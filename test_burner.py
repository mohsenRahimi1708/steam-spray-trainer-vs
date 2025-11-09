#!/usr/bin/env python3
"""
تست سریع ماژول مشعل‌ها — بدون خطا و آماده اجرا
"""

from steamlib.burners import BurnerSystem

if __name__ == "__main__":
    # ایجاد سیستم مشعل‌ها
    burners = BurnerSystem()
    
    # نمایش وضعیت کلی
    print("✅ سیستم مشعل‌ها بارگذاری شد!")
    print(f"  • تعداد کل مشعل‌ها: {len(burners)} عدد")
    print(f"  • ظرفیت حرارتی کل: {burners.total_thermal_power_MW:.1f} MW")
    
    # نمایش ارتفاع هر طبقه
    print("\n📍 ارتفاع طبقات:")
    for layer in ['A', 'B', 'C']:
        elev = burners.get_burners_by_layer(layer)[0].elevation_m
        count = len(burners.get_burners_by_layer(layer))
        print(f"  • طبقه {layer}: {elev} متر ({count} مشعل)")
    
    # نمایش یک مشعل نمونه
    fa1 = burners.get_burner("FA1")
    print(f"\n🔍 مشعل نمونه (FA1):")
    print(f"  • موقعیت: x={fa1.x_m}m, y={fa1.y_m}m, z={fa1.elevation_m}m")
    print(f"  • ظرفیت: {fa1.thermal_power_MW} MW")