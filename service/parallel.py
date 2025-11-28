import asyncio
import base64
from PIL import Image
import io
import os

# CPU 友好型压缩
def compress_image(path, max_size=512):
    img = Image.open(path)
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def process_one(path, prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, compress_image, path)


async def process_images_parallel(prompt, paths):
    tasks = [asyncio.create_task(process_one(p, prompt)) for p in paths]
    results = await asyncio.gather(*tasks)
    # 转成结构化格式给 LLM
    return [
        {"path": os.path.basename(p), "base64": b64}
        for p, b64 in zip(paths, results)
    ]
