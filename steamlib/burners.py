# steamlib/burners.py
"""
ماژول مشعل‌های گازی — با موقعیت فضایی دقیق (x, y, z)
طبقات A (9m), B (12m), C (15m)
فاصله افقی: 2m | عمق کوره: 12m
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

@dataclass
class Burner:
    """مشعل گازی با موقعیت سه‌بعدی"""
    name: str           # مثلاً "FA1"
    layer: str          # 'A', 'B', 'C'
    position: str       # 'front' یا 'rear'
    elevation_m: float  # z — ارتفاع از پایه (متر)
    x_m: float          # x — موقعیت افقی (متر)
    y_m: float          # y — عمق (0 = دیواره جلو، 12 = دیواره عقب)
    thermal_power_MW: float = 40.0

    @property
    def thermal_power_W(self) -> float:
        return self.thermal_power_MW * 1e6

    @property
    def position_3d(self) -> Tuple[float, float, float]:
        return (self.x_m, self.y_m, self.elevation_m)

@dataclass
class BurnerSystem:
    """سیستم مشعل‌های بویلر با هندسه فضایی"""
    furnace_depth_m: float = 12.0   # فاصله دیواره جلو تا عقب
    burner_spacing_m: float = 2.0   # فاصله افقی بین مشعل‌ها
    thermal_power_per_burner_MW: float = 40.0

    def __post_init__(self):
        self._burners: List[Burner] = []
        self._create_burners()

    def _create_burners(self):
        """ایجاد 24 مشعل با موقعیت x,y,z دقیق"""
        # تعریف ارتفاع هر طبقه
        layers_elevation = {
            'A': 9.0,
            'B': 12.0,   # ✅ اصلاح شد
            'C': 15.0
        }

        # موقعیت افقی مشعل‌ها (در یک ردیف 4 تایی، با فاصله 2m)
        # فرض: مرکز کوره در x=0 → مشعل‌ها در x = -3, -1, +1, +3
        x_positions = [-3.0, -1.0, 1.0, 3.0]  # متر

        for layer, z in layers_elevation.items():
            # --- مشعل‌های جلو (y = 0) ---
            for i, x in enumerate(x_positions, 1):
                name = f"F{layer}{i}"
                self._burners.append(Burner(
                    name=name,
                    layer=layer,
                    position="front",
                    elevation_m=z,
                    x_m=x,
                    y_m=0.0,  # دیواره جلو
                    thermal_power_MW=self.thermal_power_per_burner_MW
                ))
            # --- مشعل‌های عقب (y = 12) ---
            for i, x in enumerate(x_positions, 1):
                name = f"R{layer}{i}"
                self._burners.append(Burner(
                    name=name,
                    layer=layer,
                    position="rear",
                    elevation_m=z,
                    x_m=x,
                    y_m=self.furnace_depth_m,  # دیواره عقب
                    thermal_power_MW=self.thermal_power_per_burner_MW
                ))

    @property
    def burners(self) -> List[Burner]:
        return self._burners

    @property
    def total_thermal_power_MW(self) -> float:
        return len(self._burners) * self.thermal_power_per_burner_MW

    def get_burners_by_layer(self, layer: str) -> List[Burner]:
        return [b for b in self._burners if b.layer == layer]

    def get_burner(self, name: str) -> Burner:
        for burner in self._burners:
            if burner.name == name:
                return burner
        raise ValueError(f"Burner '{name}' not found")

    def distance_to_point(self, burner_name: str, point: Tuple[float, float, float]) -> float:
        """فاصله اقلیدسی بین یک مشعل و یک نقطه (مثلاً مرکز لوله سوپرهیتر)"""
        burner = self.get_burner(burner_name)
        bx, by, bz = burner.position_3d
        px, py, pz = point
        return np.sqrt((bx-px)**2 + (by-py)**2 + (bz-pz)**2)

    def view_factor_estimate(self, burner_name: str, target_point: Tuple[float, float, float]) -> float:
        """
        تخمین ساده ضریب دید (View Factor) بر اساس زاویه و فاصله
        (برای شبیه‌سازی آموزشی — نه محاسبه دقیق CFD)
        """
        d = self.distance_to_point(burner_name, target_point)
        if d < 1e-3:
            return 1.0
        # تقریب: ضریب دید ∝ 1/d² و وابسته به زاویه (در اینجا ساده‌شده)
        return min(1.0, 10.0 / (d ** 2))

    def __len__(self):
        return len(self._burners)

    def __repr__(self):
        return f"BurnerSystem(layers=A:{9}m,B:{12}m,C:{15}m, total={len(self)} burners, {self.total_thermal_power_MW} MW)"