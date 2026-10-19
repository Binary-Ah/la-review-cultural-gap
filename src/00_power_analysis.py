"""
00_power_analysis.py

"한국어 리뷰 50건 이상"이라는 기준을 감으로 정하지 않기 위한 스크립트.

두 독립집단(한국어 리뷰 감성점수 vs 영어 리뷰 감성점수) 간
Mann-Whitney U / t-test로 유의미한 차이를 검출하려면
최소 몇 건의 표본이 필요한지를, 원하는 효과크기(Cohen's d)와
검정력(power), 유의수준(alpha) 기준으로 역산한다.

사용법:
  python 00_power_analysis.py
"""
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()

ALPHA = 0.05
POWER = 0.8
EFFECT_SIZES = {
    "작은 효과 (d=0.2)": 0.2,
    "작은~중간 효과 (d=0.3)": 0.3,
    "중간 효과 (d=0.5)": 0.5,
    "큰 효과 (d=0.8)": 0.8,
}


def required_n_per_group(effect_size, alpha=ALPHA, power=POWER):
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha,
                              power=power, ratio=1.0, alternative="two-sided")
    return int(round(n))


def min_n_to_detect_given_threshold(threshold_n, alpha=ALPHA, power=POWER):
    """반대 방향: 표본 수를 고정했을 때 검출 가능한 최소 효과크기는?"""
    d = analysis.solve_power(nobs1=threshold_n, alpha=alpha,
                              power=power, ratio=1.0, alternative="two-sided")
    return d


def main():
    print(f"=== Power Analysis (alpha={ALPHA}, power={POWER}, two-sided t-test 기준) ===\n")
    print("효과크기별 집단당 최소 필요 표본 수:")
    for label, d in EFFECT_SIZES.items():
        n = required_n_per_group(d)
        print(f"  {label:25s} -> 집단당 {n:4d}건 필요")

    print()
    for threshold in [30, 50, 100]:
        d = min_n_to_detect_given_threshold(threshold)
        print(f"현재 후보 기준 min-ko-reviews={threshold}건일 때, "
              f"검출 가능한 최소 효과크기: d = {d:.3f}")

    print()
    print("[해석 가이드]")
    print("  - d < 0.2 : 사실상 무의미한 차이도 유의하다고 나올 수 있어 과도하게 민감")
    print("  - d ~ 0.3~0.5 : 데이터사이언스/설문 연구에서 흔히 의미 있다고 보는 범위")
    print("  - 최종 리포트에는 이 표를 넣고 근거를 명시")


if __name__ == "__main__":
    main()
