# steamlib/final_superheater.py

from dataclasses import dataclass
import numpy as np

@dataclass
class FinalSuperheater:
    # === مشخصات ساختاری ===
    tubes_per_tube_assembly: int = 4   # هر تیوپ = 4 لوله مشابه
    tube_assemblies_per_coil: int = 43  # هر کویل = 43 تیوپ
    tube_length_m: float = 72.0         # طول هر لوله (در هر تیوپ)

    # === مشخصات هندسی لوله ===
    tube_od_mm: float = 57.0
    tube_thickness_mm: float = 8.0
    tube_id_mm: float = 41.0

    # === موقعیت عمودی ===
    inlet_header_elevation_m: float = 40.0
    outlet_header_elevation_m: float = 42.0
    height_difference_m: float = 2.0

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

# === مشخصات ساختاری (برای تست/مستند) ===
FINAL_SPEC = {
    "structure": {
        "tubes_per_tube_assembly": 4,
        "tube_assemblies_per_coil": 43,
        "total_tube_count": 172,
        "tube_length_m": 72.0,
        "total_tube_length_m": 12384.0  # 172 × 72
    },
    "elevation": {
        "inlet_header_elevation_m": 40.0,
        "outlet_header_elevation_m": 42.0,
        "height_difference_m": 2.0
    },
    "sections": [
        {"name": "casing_1", "length_m": 1.0, "material": "12Cr1MoV"},
        {"name": "casing_2", "length_m": 2.0, "material": "12Cr2MoWVTiB"},
        {"name": "radiant", "length_m": 18.0, "material": "SA-213T91"},
        {"name": "upper", "length_m": 20.0, "material": "12Cr2MoWVTiB"},
        {"name": "lower", "length_m": 30.0, "material": "12Cr2MoWVTiB"},
        {"name": "casing_3", "length_m": 1.0, "material": "12Cr1MoV"}
    ],
    "dynamic_model": {
        "transport_delay_s": 1.8,
        "thermal_time_constant_s": 1.39
    }
}