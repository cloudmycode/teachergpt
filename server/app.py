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
import os
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

from script.build_timeline import main as build_timeline
from script.build_pptx import main as build_pptx

app = FastAPI(title="老师讲义生成", version="1.0.0")

# 任务存储（简单版，生产环境用 Redis）
tasks = {}
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 请求/响应模型 ----------
class GenerateRequest(BaseModel):
    lesson: str  # 课文名，如 "木兰词"


class TaskInfo(BaseModel):
    task_id: str
    status: str  # pending / running / success / error
    message: str
    pptx_path: Optional[str] = None
    created_at: float


# ---------- 路由 ----------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = SCRIPT_DIR / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


def _find_existing_pptx(lesson: str) -> Optional[str]:
    """查找已存在的 PPT 文件"""
    from script.generate import lesson_name_to_pinyin
    lesson_dir_name = lesson_name_to_pinyin(lesson)
    lesson_dir = PROJECT_ROOT / "data" / "timelines" / lesson_dir_name
    if lesson_dir.exists():
        for f in lesson_dir.glob("*.pptx"):
            return str(f)
    return None


@app.post("/api/generate", response_model=TaskInfo)
async def generate_pptx(request: GenerateRequest, background_tasks: BackgroundTasks):
    """提交生成任务，返回 task_id"""
    # 先检查是否已存在 PPT
    existing_pptx = _find_existing_pptx(request.lesson)
    if existing_pptx:
        task_id = str(uuid.uuid4())[:8]
        tasks[task_id] = {
            "status": "success",
            "message": "已找到现有文件",
            "pptx_path": existing_pptx,
            "created_at": time.time(),
            "error": None,
        }
        return TaskInfo(
            task_id=task_id,
            status="success",
            message="已找到现有文件，可直接下载",
            pptx_path=existing_pptx,
            created_at=tasks[task_id]["created_at"],
        )

    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {
        "status": "pending",
        "message": "任务已提交",
        "pptx_path": None,
        "created_at": time.time(),
        "error": None,
    }

    background_tasks.add_task(_run_generation, task_id, request)

    return TaskInfo(
        task_id=task_id,
        status="pending",
        message="任务已提交，正在获取原文并智能分批生成...",
        created_at=tasks[task_id]["created_at"],
    )


@app.get("/api/task/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """查询任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = tasks[task_id]
    return TaskInfo(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        pptx_path=task.get("pptx_path"),
        created_at=task["created_at"],
    )


@app.get("/api/download/{task_id}")
async def download_pptx(task_id: str):
    """下载生成的 PPT"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = tasks[task_id]
    if task["status"] != "success" or not task.get("pptx_path"):
        raise HTTPException(status_code=400, detail=f"PPT 未生成完成: {task['message']}")

    pptx_path = Path(task["pptx_path"])
    if not pptx_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        path=str(pptx_path),
        filename=pptx_path.name,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


# ---------- 后台任务 ----------
async def _run_generation(task_id: str, request: GenerateRequest):
    """后台执行生成流程"""
    task = tasks[task_id]
    lesson = request.lesson

    try:
        task["status"] = "running"
        task["message"] = f"正在获取《{lesson}》原文并分析分批策略..."

        # 1. 生成时间轴（含 TTS）
        # 后端会自动根据原文长度决定分批策略
        timeline_args = [f"{lesson} 全文"]

        # 运行 build_timeline（同步）
        await asyncio.to_thread(_run_build_timeline, timeline_args)

        task["message"] = "讲稿和音频生成完成，正在生成 PPT..."

        # 2. 生成 PPT
        pptx_path = await asyncio.to_thread(_run_build_pptx, [lesson])

        task["status"] = "success"
        task["message"] = "生成完成！"
        task["pptx_path"] = str(pptx_path)

    except Exception as e:
        task["status"] = "error"
        task["message"] = f"生成失败: {str(e)}"
        task["error"] = str(e)


def _run_build_timeline(args: list) -> None:
    """同步运行 build_timeline"""
    import subprocess
    script_path = PROJECT_ROOT / "script" / "build_timeline.py"
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"build_timeline 失败: {result.stderr or result.stdout}")


def _run_build_pptx(args: list) -> str:
    """同步运行 build_pptx，返回 PPT 路径"""
    import subprocess
    script_path = PROJECT_ROOT / "script" / "build_pptx.py"
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"build_pptx 失败: {result.stderr or result.stdout}")

    # 找到生成的 pptx 文件（目录名使用拼音）
    from script.generate import lesson_name_to_pinyin
    lesson_clean = args[0].split()[0]  # 取课文名（去掉"全文"等）
    lesson_dir_name = lesson_name_to_pinyin(lesson_clean)
    pptx_dir = PROJECT_ROOT / "data" / "timelines" / lesson_dir_name
    for f in pptx_dir.glob("*.pptx"):
        return str(f)
    raise FileNotFoundError(f"未找到生成的 PPT 文件: {pptx_dir}")


# ---------- 启动 ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
