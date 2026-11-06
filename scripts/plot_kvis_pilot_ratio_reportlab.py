from __future__ import annotations

"""Render the completed pilot risk-adjusted-ratio figure without Matplotlib."""

from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white


OUTPUT = Path("reports/figures/kvis_pilot_validation_sharpe_2855.pdf")


def main() -> None:
    width, height = 720.0, 430.0
    output = OUTPUT.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    ink = HexColor("#1F2937")
    muted = HexColor("#4B5563")
    grid = HexColor("#D1D5DB")
    blue = HexColor("#2563EB")
    gold = HexColor("#D97706")
    light = HexColor("#E5E7EB")

    rows = [
        ("PTT buy-and-hold", 0.405480, gold),
        ("ADVANC buy-and-hold", 0.097452, gold),
        ("Equal weight", -0.416554, light),
        ("A2C", -0.434028, blue),
        ("PPO", -0.439532, blue),
        ("Mean variance", -1.097376, light),
    ]

    chart_left, chart_right = 220.0, 670.0
    chart_bottom, chart_top = 80.0, 330.0
    x_min, x_max = -1.2, 0.5

    def x_pos(value: float) -> float:
        return chart_left + (value - x_min) / (x_max - x_min) * (chart_right - chart_left)

    pdf = canvas.Canvas(str(output), pagesize=(width, height))
    pdf.setTitle("Real-OHLCV pilot validation risk-adjusted ratio")
    pdf.setAuthor("Nathawat Youngyim; Kunach Kijsanayothin")
    pdf.setFillColor(white)
    pdf.rect(0, 0, width, height, stroke=0, fill=1)

    pdf.setFillColor(ink)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(62, 392, "Real-OHLCV pilot validation risk-adjusted ratio")
    pdf.setFillColor(muted)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(62, 373, "Frozen May 2026 table; compounded annualized return divided by annualized volatility")

    ticks = [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
    for tick in ticks:
        x = x_pos(tick)
        pdf.setStrokeColor(ink if tick == 0 else grid)
        pdf.setLineWidth(1.2 if tick == 0 else 0.6)
        pdf.line(x, chart_bottom, x, chart_top)
        pdf.setFillColor(ink)
        pdf.setFont("Helvetica", 8)
        pdf.drawCentredString(x, chart_bottom - 14, f"{tick:.1f}")

    y_start, y_step, bar_height = 308.0, 42.0, 24.0
    zero_x = x_pos(0.0)
    for index, (label, value, color) in enumerate(rows):
        y = y_start - index * y_step
        end_x = x_pos(value)
        left = min(zero_x, end_x)
        bar_width = abs(end_x - zero_x)
        pdf.setFillColor(color)
        pdf.setStrokeColor(ink)
        pdf.setLineWidth(0.8)
        pdf.rect(left, y - bar_height / 2, bar_width, bar_height, stroke=1, fill=1)

        pdf.setFillColor(ink)
        pdf.setFont("Helvetica", 9)
        pdf.drawRightString(chart_left - 14, y - 3, label)

        if value >= 0:
            value_x = end_x + 8
            pdf.setFillColor(ink)
            pdf.drawString(value_x, y - 3, f"{value:+.6f}")
        else:
            value_x = end_x + 8
            pdf.setFillColor(white if color == blue else ink)
            pdf.setFont("Helvetica-Bold", 9)
            pdf.drawString(value_x, y - 3, f"{value:+.6f}")

    pdf.setFillColor(ink)
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString((chart_left + chart_right) / 2, 43, "Reported annualized return-to-volatility ratio")
    pdf.setFillColor(muted)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(62, 18, "PPO and A2C did not outperform the simple baselines in this completed validation pilot.")

    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
