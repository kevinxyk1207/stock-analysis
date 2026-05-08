"""
精筛过滤器 v1：从扫描器候选池中自动过滤
Layer 1: 财务质量(利润真实/现金流/利润率)
Layer 2: 行业聚焦(排除纯周期/农业/消费)
输出≤30只 → Claude手工精筛 → ≤5只入精筛池
"""
import sys, os, json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

CANDIDATES = "data_cache/scanner_candidates.json"
OUTPUT = "data_cache/refined_candidates.json"

# Claude已淘汰的股票（手动维护）
REJECTED = {"688111", "002582", "000506", "000426", "002653", "300723", "688266",
            "688221", "300390", "002192", "002842", "603045", "300970", "002297",
            "301500", "301246", "301292", "002709", "000973", "300511"}

# 行业白名单（包含关键词即通过）
TECH_KW = ["半导体","芯片","电子","计算机","通信","软件","光模块","光纤",
           "PCB","铜箔","靶材","算力","服务器","液冷","电源","连接器",
           "存储","MCU","SoC","FPGA","射频","传感器","机器人",
           "材料","设备","制造","新能源","汽车","锂","储能","钨","稀土"]

# 行业黑名单（包含即排除）
JUNK_KW = ["农业","养殖","食品","饮料","珠宝","零售","餐饮","房地产",
           "建筑","装饰","传媒","广告","教育","旅游","纺织","服装",
           "造纸","家具","包装","钢铁","煤炭","石油"]


def load_candidates():
    with open(CANDIDATES, encoding="utf-8") as f:
        return json.load(f)["candidates"]


def refine(pool):
    """双Layer过滤"""
    # Layer 1: 财务质量
    layer1 = []
    for r in pool:
        rev = r.get("Q1营收(亿)") or 0
        rev_g = r.get("营收增速%") or 0
        prf_g = r.get("利润增速%") or 0
        margin = r.get("毛利率%") or 0

        if rev < 2: continue                    # 营收太小不稳定
        if margin < 15: continue                # 毛利太薄
        if prf_g > 500 and rev_g < 10: continue # 一次性收益嫌疑
        layer1.append(r)

    # Layer 2: 行业聚焦
    layer2 = []
    for r in layer1:
        ind = str(r.get("行业") or "")
        # 黑名单
        junk = False
        for kw in JUNK_KW:
            if kw in ind:
                junk = True; break
        if junk: continue
        # 白名单
        tech = False
        for kw in TECH_KW:
            if kw in ind:
                tech = True; break
        if tech:
            layer2.append(r)

    # Layer 3: Claude已淘汰 + 已有池去重
    layer3 = [r for r in layer2 if r["代码"] not in REJECTED]

    early = set()
    for f in ["data_cache/early_watch.json", "data_cache/curated_pool.json"]:
        if os.path.exists(f):
            with open(f) as fp:
                d = json.load(fp)
                for s in d.get("stocks", []):
                    early.add(s["symbol"])

    layer3 = [r for r in layer3 if r["代码"] not in early]

    return layer3


if __name__ == "__main__":
    pool = load_candidates()
    print(f"原始候选池: {len(pool)}只")

    refined = refine(pool)
    print(f"Layer1(财务): {len([r for r in pool if (r.get('Q1营收(亿)')or 0)>=2 and (r.get('毛利率%')or 0)>=15])}只")
    print(f"Layer2(行业): {len(refined)}只")

    # 保存
    os.makedirs("data_cache", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d"), "candidates": refined}, f, ensure_ascii=False, indent=2)

    # 打印Top 15
    print(f"\n精筛候选 Top 15:")
    for i, r in enumerate(refined[:15]):
        print(f"{i+1:>2}. {r['代码']} {r['名称']:<8} {r['行业']:<20} 利润{r['利润增速%']:>+6.0f}% 毛利{r['毛利率%']:.0f}% PE{r.get('研报共识PE') or '—'}")
    print(f"\n已保存: {OUTPUT}")
