# steamlib/platen_superheater.py

from dataclasses import dataclass
import numpy as np

@dataclass
class PlatenSuperheater:
    # === ساختار ===
    tubes_per_tube_assembly: int = 4    # هر تیوپ = 4 لوله مشابه
    tube_assemblies_per_coil: int = 43   # هر کویل = 43 تیوپ
    tube_length_m: float = 50.0          # طول هر لوله (همه لوله‌ها یکسان)

    # === هندسه لوله ===
    tube_od_mm: float = 57.0
    tube_thickness_mm: float = 8.0
    tube_id_mm: float = 41.0

    # === موقعیت عمودی (از نقشه) ===
    inlet_header_elevation_m: float = 34.774  # هدر بالا = ورودی
    outlet_header_elevation_m: float = 31.880  # هدر پایین = خروجی
    height_difference_m: float = 2.894

    def __post_init__(self):
        self.cross_section_m2 = np.pi * (self.tube_id_mm / 2000) ** 2
        self.total_tube_count = self.tube_assemblies_per_coil * self.tubes_per_tube_assembly
        self.total_tube_length_m = self.total_tube_count * self.tube_length_m
        self.outer_surface_m2 = np.pi * (self.tube_od_mm / 1000) * self.total_tube_length_m

    @property
    def steam_volume_m3(self) -> float:
        return self.total_tube_length_m * self.cross_section_m2

    @property
    def steam_mass_kg(self, rho=35.0) -> float:
        return rho * self.steam_volume_m3

    def transport_delay_s(self, v_steam_mps=40.0) -> float:
        return self.tube_length_m / v_steam_mps

    def thermal_time_constant_s(self, U=1200, cp_steam=4300, rho_steam=35.0) -> float:
        C_th = self.steam_mass_kg(cp_steam) * cp_steam
        return C_th / (U * self.outer_surface_m2)

    def foptd_step_response(self, t, K=1.0):
        theta = self.transport_delay_s()
        tau = self.thermal_time_constant_s()
        y = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti > theta:
                y[i] = K * (1 - np.exp(-(ti - theta) / tau))
        return y

# === مشخصات بخش‌های لوله (برای مستند/شبیه‌سازی رنگی) ===
PLATEN_TUBE_SECTIONS = [
    {"name": "inlet_casing", "color": "red", "length_m": 2.5, "material": "12Cr1MoV", "position": "cabinet_casing"},
    {"name": "upper_horizontal", "color": "green", "length_m": 12.0, "material": "12Cr2MoWVTiB", "position": "boiler"},
    {"name": "lower_vertical", "color": "orange", "length_m": 20.0, "material": "12Cr2MoWVTiB", "position": "boiler"},
    {"name": "radiant_section", "color": "purple", "length_m": 15.0, "material": "SA-213T91", "position": "boiler"},
    {"name": "outlet_casing", "color": "yellow", "length_m": 0.5, "material": "12Cr1MoV", "position": "cabinet_casing"}
]

# === مشخصات کلی (برای YAML/تست) ===
PLATEN_SPEC = {
    "structure": {
        "tubes_per_tube_assembly": 4,
        "tube_assemblies_per_coil": 43,
        "total_tube_count": 172,
        "tube_length_m": 50.0,
        "total_tube_length_m": 8600.0  # 172 × 50
    },
    "elevation": {
        "inlet_header_elevation_m": 34.774,
        "outlet_header_elevation_m": 31.880,
        "height_difference_m": 2.894
    },
    "sections": PLATEN_TUBE_SECTIONS,
    "thermal": {
        "steam_density_kg_m3": 35.0,
        "steam_cp_J_kgK": 4300,
        "total_steam_mass_kg": 397.32,
        "total_thermal_capacitance_J_K": 1.708e6
    },
    "dynamic_model": {
        "transport_delay_s": 1.25,     # 50 / 40
        "thermal_time_constant_s": 1.15,
        "transfer_function": "K * exp(-1.25*s) / (1.15*s + 1)"
    }
}