@echo off
chcp 65001 >nul
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [错误] 未找到 Python 虚拟环境: %VENV_PY%
    pause
    exit /b 1
)
echo ========================================
echo   Cookie 获取工具 - 今日头条自动发文 (Edge)
echo ========================================
echo.
echo 即将打开 Edge 浏览器，请登录 mp.toutiao.com
echo 登录完成后回到这里按回车键
echo.
pause

"%VENV_PY%" -c "
import asyncio, os
os.environ['DEEPSEEK_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
from toutiao import *

async def capture():
    playwright = await async_playwright().start()
    # 使用系统安装的 Edge 浏览器
    browser = await playwright.chromium.launch(
        headless=False,
        channel='msedge'
    )
    context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
    page = await context.new_page()
    print('请在 Edge 浏览器中登录今日头条创作者平台...')
    await page.goto('https://mp.toutiao.com')
    input('登录完成后按回车继续...')
    cookies = await context.cookies()
    user_agent = await page.evaluate('() => navigator.userAgent')
    fingerprint = {
        'viewport': '1920,1080',
        'locale': await page.evaluate('() => navigator.language'),
        'timezone': await page.evaluate('() => Intl.DateTimeFormat().resolvedOptions().timeZone'),
        'platform': await page.evaluate('() => navigator.platform')
    }
    cookie_pool = CookiePool()
    cookie_pool.save_account('account_01', cookies, user_agent, fingerprint)
    print(f'account_01 Cookie 已保存！共 {len(cookies)} 条')
    await browser.close()

asyncio.run(capture())
"
pause
