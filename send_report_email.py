"""
send_report_email.py
发送最新选股报告到邮箱
"""
import os
import glob
import smtplib
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# ── 邮件配置 ──
SMTP_HOST = "smtp.163.com"
SMTP_PORT = 465
SENDER    = "youxiang020766@163.com"
PASSWORD  = os.environ.get("EMAIL_PASSWORD", "")   # 从环境变量读取授权码
RECEIVER  = "youxiang020766@163.com"

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def find_latest_csv():
    """找到最新的选股结果文件（支持 csv 和 xlsx）"""
    csv_files  = glob.glob(os.path.join(REPORTS_DIR, "screening_*.csv"))
    xlsx_files = glob.glob(os.path.join(REPORTS_DIR, "screening_*.xlsx"))
    files = sorted(csv_files + xlsx_files, reverse=True)
    if not files:
        raise FileNotFoundError("未找到选股结果文件，请先运行 daily_update.py")
    return files[0]


def get_market_regime_str():
    """动态获取市场环境描述"""
    try:
        from enhanced_fetcher import load_cache_data, EnhancedStockFetcher
        import pandas as pd
        fetcher = EnhancedStockFetcher()
        hs300 = set(str(s).zfill(6) for s in fetcher.get_hs300_stocks()['code'].tolist())
        raw = load_cache_data(min_rows=200, common_range=False)
        stock_data = {k: v for k, v in raw.items() if str(k).zfill(6) in hs300}
        close_dict = {sym: pd.Series(df['close'].values) for sym, df in stock_data.items()}
        mc = pd.DataFrame(close_dict).mean(axis=1)
        mma60 = mc.rolling(60, min_periods=60).mean()
        if len(mma60) > 0 and pd.notna(mma60.iloc[-1]):
            is_bull = mc.iloc[-1] > mma60.iloc[-1]
            color = 'red' if is_bull else 'green'
            label = '牛市' if is_bull else '熊市'
            return f'<b style=\"color:{color}\">{label}</b>（均价 {mc.iloc[-1]:.2f} vs MA60 {mma60.iloc[-1]:.2f}）'
    except Exception:
        pass
    return '<b>未知</b>'


def build_html_table(df_short, df_long):
    """生成HTML格式的选股表格"""
    def df_to_html(df):
        rows = ""
        for _, r in df.iterrows():
            rows += (
                f"<tr>"
                f"<td>{int(r['排名'])}</td>"
                f"<td style='font-family:monospace'>{str(r['代码']).zfill(6)}</td>"
                f"<td>{r['名称']}</td>"
                f"<td>{r['收盘']:.2f}</td>"
                f"<td>{r['J值']:.1f}</td>"
                f"<td>{r['zxdkx比']:.3f}</td>"
                f"<td><b>{r['得分']:.1f}</b></td>"
                f"</tr>"
            )
        return rows

    th = "<th style='padding:6px 10px;background:#2c3e50;color:white'>%s</th>"
    header = "".join(th % h for h in ["排名", "代码", "名称", "收盘", "J值", "zxdkx比", "得分"])

    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#333">
    <h2 style="color:#2c3e50">B1 选股报告 — {datetime.now().strftime('%Y-%m-%d')}</h2>
    <p>市场环境：{get_market_regime_str()}</p>

    <h3 style="color:#e74c3c">短线 Top 10</h3>
    <table border="0" cellspacing="0" cellpadding="4"
           style="border-collapse:collapse;width:600px">
      <thead><tr>{header}</tr></thead>
      <tbody style="background:#fafafa">{df_to_html(df_short)}</tbody>
    </table>

    <h3 style="color:#2980b9" style="margin-top:24px">长线 Top 10</h3>
    <table border="0" cellspacing="0" cellpadding="4"
           style="border-collapse:collapse;width:600px">
      <thead><tr>{header}</tr></thead>
      <tbody style="background:#fafafa">{df_to_html(df_long)}</tbody>
    </table>

    <p style="color:#999;font-size:12px;margin-top:20px">
      本邮件由自动化程序发送，数据基于 {datetime.now().strftime('%Y-%m-%d')} 收盘数据。
    </p>
    </body></html>
    """
    return html


def send_email():
    if not PASSWORD:
        raise ValueError(
            "未设置邮箱授权码！请设置环境变量 EMAIL_PASSWORD\n"
            "例如：$env:EMAIL_PASSWORD='你的163邮箱授权码'"
        )

    csv_path = find_latest_csv()
    print(f"读取结果文件: {csv_path}")

    if csv_path.endswith(".xlsx"):
        df = pd.read_excel(csv_path, dtype={"代码": str})
    else:
        df = pd.read_csv(csv_path, dtype={"代码": str})
    df_short = df[df["期限"] == "短线"].head(10).reset_index(drop=True)
    df_long  = df[df["期限"] == "长线"].head(10).reset_index(drop=True)

    # 构建邮件
    msg = MIMEMultipart("alternative")
    today = datetime.now().strftime("%Y-%m-%d")
    msg["Subject"] = f"B1选股报告 {today}"
    msg["From"]    = SENDER
    msg["To"]      = RECEIVER

    html_content = build_html_table(df_short, df_long)
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    # 附加结果文件
    with open(csv_path, "rb") as f:
        mime_subtype = "vnd.openxmlformats-officedocument.spreadsheetml.sheet" if csv_path.endswith(".xlsx") else "octet-stream"
        part = MIMEBase("application", mime_subtype)
        part.set_payload(f.read())
        encoders.encode_base64(part)
        fname = os.path.basename(csv_path)
        part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
        msg.attach(part)

    # 发送
    print(f"正在发送邮件到 {RECEIVER} ...")
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SENDER, PASSWORD)
        server.sendmail(SENDER, RECEIVER, msg.as_string())

    print(f"邮件发送成功！报告日期: {today}")


if __name__ == "__main__":
    send_email()
