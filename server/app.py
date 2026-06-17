#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
老师讲义生成 Web 服务

启动：
  cd /Users/wang/Project/teachergpt
  python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000

访问：
  http://localhost:8000
"""

import asyncio
import re
import secrets
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR / "script"))

app = FastAPI(title="老师讲义生成", version="1.0.0")

# 任务持久化。MVP 用 SQLite，后续并发量上来再换 Redis/Postgres。
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = OUTPUT_DIR / "tasks.sqlite3"
TIMELINE_DIR = PROJECT_ROOT / "data" / "timelines"
TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/timelines", StaticFiles(directory=str(TIMELINE_DIR)), name="timelines")


# ---------- 请求/响应模型 ----------
class GenerateRequest(BaseModel):
    lesson: str  # 用户输入：课文名/章节范围/短文章内容
    client_id: Optional[str] = None


class TaskInfo(BaseModel):
    task_id: Optional[str] = None
    token: Optional[str] = None
    status: str  # pending / running / success / error / need_scope
    message: str
    pptx_path: Optional[str] = None
    download_url: Optional[str] = None
    preview_url: Optional[str] = None
    recover_url: Optional[str] = None
    created_at: float


# ---------- 路由 ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = SCRIPT_DIR / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                token TEXT NOT NULL,
                client_id TEXT,
                lesson TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                pptx_path TEXT,
                preview_path TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_client ON tasks(client_id, created_at)")


_init_db()


def _task_to_info(row: sqlite3.Row, include_token: bool = False) -> TaskInfo:
    token = row["token"] if include_token else None
    task_id = row["task_id"]
    preview_url = None
    download_url = None
    recover_url = None
    if row["status"] == "success":
        preview_url = f"/api/preview/{task_id}?token={row['token']}"
        download_url = f"/api/download/{task_id}?token={row['token']}"
    if include_token:
        recover_url = f"/task/{task_id}?token={row['token']}"
    return TaskInfo(
        task_id=task_id,
        token=token,
        status=row["status"],
        message=row["message"],
        pptx_path=row["pptx_path"],
        preview_url=preview_url,
        download_url=download_url,
        recover_url=recover_url,
        created_at=row["created_at"],
    )


def _get_task(task_id: str) -> sqlite3.Row:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="任务不存在")
    return row


def _check_token(row: sqlite3.Row, token: Optional[str]) -> None:
    if not token or not secrets.compare_digest(token, row["token"]):
        raise HTTPException(status_code=403, detail="任务链接无效")


def _create_task(lesson: str, client_id: Optional[str], status: str, message: str,
                 pptx_path: Optional[str] = None, preview_path: Optional[str] = None) -> sqlite3.Row:
    task_id = uuid.uuid4().hex
    token = secrets.token_urlsafe(24)
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO tasks (
                task_id, token, client_id, lesson, status, message,
                pptx_path, preview_path, error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (task_id, token, client_id, lesson, status, message, pptx_path, preview_path, now, now),
        )
    return _get_task(task_id)


def _update_task(task_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    keys = list(fields.keys())
    sql = ", ".join(f"{key} = ?" for key in keys)
    values = [fields[key] for key in keys]
    values.append(task_id)
    with _connect() as conn:
        conn.execute(f"UPDATE tasks SET {sql} WHERE task_id = ?", values)


def _input_needs_scope(text: str) -> Optional[str]:
    text = text.strip()
    has_scope = re.search(r"第\s*[\d一二三四五六七八九十百]+[\s\-到至、,，\d一二三四五六七八九十百]*\s*(段|章|节|则|句|页)", text)
    looks_like_article = len(text) > 220 or text.count("\n") >= 4 or len(re.findall(r"[，。！？；：]", text)) >= 18
    if looks_like_article and not has_scope:
        return (
            "输入内容较长，整篇生成会很慢。请具体到段落或章节，例如："
            "《赤壁赋》第一段、 《出师表》第1-3段、 《世说新语·德行篇》第25则。"
        )
    return None


def _has_scope(text: str) -> bool:
    return bool(re.search(r"第\s*[\d一二三四五六七八九十百]+[\s\-到至、,，\d一二三四五六七八九十百]*\s*(段|章|节|则|句|页)", text))


def _build_generation_query(text: str) -> str:
    return text if _has_scope(text) else f"{text} 全文"


def _extract_lesson_name(query: str) -> str:
    from script import generate
    return generate._extract_lesson(query)


def _find_existing_pptx(lesson: str) -> Optional[str]:
    """查找已存在的 PPT 文件"""
    from script.generate import lesson_name_to_pinyin
    lesson_dir_name = lesson_name_to_pinyin(_extract_lesson_name(lesson))
    lesson_dir = PROJECT_ROOT / "data" / "timelines" / lesson_dir_name
    if lesson_dir.exists():
        for f in lesson_dir.glob("*.pptx"):
            return str(f)
    return None


@app.post("/api/generate", response_model=TaskInfo)
async def generate_pptx(request: GenerateRequest, background_tasks: BackgroundTasks):
    """提交生成任务，返回 task_id"""
    lesson = request.lesson.strip()
    if not lesson:
        raise HTTPException(status_code=400, detail="请输入课文名或段落范围")

    scope_message = _input_needs_scope(lesson)
    if scope_message:
        return TaskInfo(
            status="need_scope",
            message=scope_message,
            created_at=time.time(),
        )

    # 先检查是否已存在 PPT
    existing_pptx = _find_existing_pptx(lesson)
    if existing_pptx:
        preview_path = _find_existing_preview(lesson)
        row = _create_task(
            lesson=lesson,
            client_id=request.client_id,
            status="success",
            message="已找到现有文件，可直接查看",
            pptx_path=existing_pptx,
            preview_path=preview_path,
        )
        return _task_to_info(row, include_token=True)

    row = _create_task(
        lesson=lesson,
        client_id=request.client_id,
        status="pending",
        message="任务已提交，正在排队生成...",
    )

    background_tasks.add_task(_run_generation, row["task_id"])

    return _task_to_info(row, include_token=True)


@app.get("/api/task/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str, token: Optional[str] = None):
    """查询任务状态"""
    row = _get_task(task_id)
    _check_token(row, token)
    return _task_to_info(row, include_token=True)


@app.get("/api/tasks", response_model=list[TaskInfo])
async def list_tasks(client_id: str):
    """按匿名 client_id 恢复最近任务"""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM tasks
            WHERE client_id = ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (client_id,),
        ).fetchall()
    return [_task_to_info(row, include_token=True) for row in rows]


@app.get("/api/download/{task_id}")
async def download_pptx(task_id: str, token: Optional[str] = None):
    """下载生成的 PPT"""
    task = _get_task(task_id)
    _check_token(task, token)
    if task["status"] != "success" or not task["pptx_path"]:
        raise HTTPException(status_code=400, detail=f"PPT 未生成完成: {task['message']}")

    pptx_path = Path(task["pptx_path"])
    if not pptx_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        path=str(pptx_path),
        filename=pptx_path.name,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


@app.get("/api/preview/{task_id}", response_class=HTMLResponse)
async def preview_player(task_id: str, token: Optional[str] = None):
    """展示生成好的 HTML 预览"""
    task = _get_task(task_id)
    _check_token(task, token)
    if task["status"] != "success" or not task["preview_path"]:
        raise HTTPException(status_code=400, detail=f"预览未生成完成: {task['message']}")
    preview_path = Path(task["preview_path"])
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="预览文件不存在")

    html = preview_path.read_text(encoding="utf-8")
    try:
        rel_dir = preview_path.parent.relative_to(TIMELINE_DIR)
        base_href = f"/timelines/{rel_dir.as_posix()}/"
        html = html.replace("<head>", f'<head>\n<base href="{base_href}">', 1)
    except ValueError:
        pass
    return HTMLResponse(html)


@app.get("/task/{task_id}", response_class=HTMLResponse)
async def recover_page(task_id: str, token: str):
    """恢复链接入口：仍然返回单页应用，让前端按 URL 恢复任务"""
    html_path = SCRIPT_DIR / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


# ---------- 后台任务 ----------
async def _run_generation(task_id: str):
    """后台执行生成流程"""
    task = _get_task(task_id)
    lesson = task["lesson"]

    try:
        _update_task(
            task_id,
            status="running",
            message=f"正在获取《{lesson}》原文并分析分批策略...",
        )

        # 1. 一站式生成：讲稿 + 音频 + PPT（buildclass）
        # 后端会自动根据原文长度决定分批策略
        # 传 --lesson 明确指定文章名，确保目录名正确
        build_args = [
            _build_generation_query(lesson), "--lesson", lesson,
            "--audio", "--pptx",
        ]

        # 运行 buildclass（同步）
        pptx_path = await asyncio.to_thread(_run_buildclass, build_args)

        _update_task(task_id, message="PPT 生成完成，正在生成网页预览...")
        preview_path = await asyncio.to_thread(_run_build_player, [lesson])

        _update_task(
            task_id,
            status="success",
            message="生成完成！",
            pptx_path=str(pptx_path),
            preview_path=str(preview_path),
        )

    except Exception as e:
        _update_task(
            task_id,
            status="error",
            message=f"生成失败: {str(e)}",
            error=str(e),
        )


def _run_buildclass(args: list) -> str:
    """同步运行 buildclass（讲稿 + 音频 + PPT），返回 PPT 路径"""
    import subprocess
    script_path = PROJECT_ROOT / "script" / "buildclass.py"
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"buildclass 失败: {result.stderr or result.stdout}")

    # 找到生成的 pptx 文件（目录名使用拼音，可能是 课程/章节 两级）
    from script.generate import lesson_name_to_pinyin
    lesson_clean = _extract_lesson_name(args[0])
    lesson_dir_name = lesson_name_to_pinyin(lesson_clean)
    timelines_dir = PROJECT_ROOT / "data" / "timelines"
    # 优先精确目录，否则全局递归找
    candidates = list((timelines_dir / lesson_dir_name).rglob("*.pptx"))
    if not candidates:
        candidates = list(timelines_dir.rglob("*.pptx"))
    if candidates:
        return str(max(candidates, key=lambda f: f.stat().st_mtime))
    raise FileNotFoundError(f"未找到生成的 PPT 文件: {timelines_dir / lesson_dir_name}")


def _run_build_player(args: list) -> str:
    """同步运行 build_player，返回 HTML 预览路径"""
    import subprocess
    script_path = PROJECT_ROOT / "script" / "build_player.py"
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"build_player 失败: {result.stderr or result.stdout}")

    from script.generate import lesson_name_to_pinyin
    lesson_clean = _extract_lesson_name(args[0])
    lesson_dir_name = lesson_name_to_pinyin(lesson_clean)
    preview_path = PROJECT_ROOT / "data" / "timelines" / lesson_dir_name / "player.html"
    if preview_path.exists():
        return str(preview_path)
    raise FileNotFoundError(f"未找到生成的预览文件: {preview_path}")


def _find_existing_preview(lesson: str) -> Optional[str]:
    from script.generate import lesson_name_to_pinyin
    lesson_dir_name = lesson_name_to_pinyin(_extract_lesson_name(lesson))
    preview_path = PROJECT_ROOT / "data" / "timelines" / lesson_dir_name / "player.html"
    return str(preview_path) if preview_path.exists() else None


# ---------- 启动 ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
