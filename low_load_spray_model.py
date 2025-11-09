"""
مدل‌سازی دینامیکی پاسخ اسپری در شرایط راه‌اندازی (Low Load)
بویلر: 1000 TPH, 170 bar, 540°C
شرایط استارت: 100-300 TPH, 35-100 bar, 350-450°C
تاخیر اسپری: ~10 دقیقه
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List
import json

# =====================================================
# 1. محاسبه خواص بخار با IAPWS (ساده‌شده)
# =====================================================
class SteamProperties:
    """محاسبات ساده خواص بخار فوق‌گرم"""
    
    @staticmethod
    def density_kg_m3(P_bar: float, T_C: float) -> float:
        """چگالی بخار فوق‌گرم"""
        # رابطه تقریبی: ρ ≈ P/(R*T)
        P_Pa = P_bar * 1e5
        T_K = T_C + 273.15
        R_specific = 461.5  # J/(kg·K) برای بخار آب
        return P_Pa / (R_specific * T_K)
    
    @staticmethod
    def specific_heat_J_kgK(P_bar: float, T_C: float) -> float:
        """ظرفیت گرمایی ویژه بخار"""
        # تقریب خطی
        if T_C < 400:
            return 2200 + (T_C - 300) * 5
        else:
            return 2700 + (T_C - 400) * 3
    
    @staticmethod
    def velocity_m_s(mass_flow_kg_s: float, P_bar: float, T_C: float, 
                     tube_id_mm: float, n_tubes: int) -> float:
        """سرعت جریان بخار در لوله"""
        rho = SteamProperties.density_kg_m3(P_bar, T_C)
        A_total = n_tubes * np.pi * (tube_id_mm/2000)**2
        return mass_flow_kg_s / (rho * A_total)

# =====================================================
# 2. مدل سوپرهیتر با پارامترهای وابسته به بار
# =====================================================
@dataclass
class SuperheaterDynamicModel:
    """مدل دینامیکی سوپرهیتر با تغییرات بار"""
    name: str
    tube_length_m: float
    tube_id_mm: float = 41.0
    n_tubes: int = 172
    
    # مشخصات انتقال حرارت
    base_U_W_m2K: float = 1200.0  # ضریب انتقال حرارت نامی
    outer_surface_m2: float = 0.0
    
    def __post_init__(self):
        tube_od_mm = 57.0
        self.outer_surface_m2 = np.pi * (tube_od_mm/1000) * \
                                self.n_tubes * self.tube_length_m
    
    def transport_delay_s(self, load_percent: float, 
                         P_bar: float, T_C: float) -> float:
        """
        تاخیر انتقالی (θ) وابسته به بار
        در بار پایین: سرعت کم → تاخیر زیاد
        """
        mass_flow = (load_percent / 100) * 278.0  # kg/s
        v_steam = SteamProperties.velocity_m_s(
            mass_flow, P_bar, T_C, self.tube_id_mm, self.n_tubes
        )
        
        # تاخیر = طول / سرعت
        delay = self.tube_length_m / max(v_steam, 1.0)
        return delay
    
    def thermal_time_constant_s(self, load_percent: float,
                                P_bar: float, T_C: float) -> float:
        """
        ثابت زمانی حرارتی (τ) وابسته به بار
        τ = (m × cp) / (U × A)
        
        در بار پایین:
        - جرم بخار کم → τ کم می‌شود
        - U کم (جابجایی ضعیف) → τ زیاد می‌شود
        - تأثیر کلی: τ معمولاً زیاد می‌شود
        """
        mass_flow = (load_percent / 100) * 278.0
        rho = SteamProperties.density_kg_m3(P_bar, T_C)
        cp = SteamProperties.specific_heat_J_kgK(P_bar, T_C)
        
        # حجم بخار در لوله‌ها
        V_m3 = self.n_tubes * np.pi * (self.tube_id_mm/2000)**2 * self.tube_length_m
        m_steam = rho * V_m3
        
        # ضریب انتقال حرارت وابسته به سرعت (Re^0.8)
        v_steam = SteamProperties.velocity_m_s(
            mass_flow, P_bar, T_C, self.tube_id_mm, self.n_tubes
        )
        velocity_factor = (v_steam / 40.0) ** 0.8  # نرمال‌شده نسبت به 40 m/s
        U_actual = self.base_U_W_m2K * max(velocity_factor, 0.3)
        
        # ظرفیت حرارتی کل
        C_thermal = m_steam * cp
        
        # ثابت زمانی
        tau = C_thermal / (U_actual * self.outer_surface_m2)
        return tau
    
    def overall_time_constant_s(self, load_percent: float,
                               P_bar: float, T_C: float) -> float:
        """ثابت زمانی کلی (τ + θ/3 تقریبی)"""
        tau = self.thermal_time_constant_s(load_percent, P_bar, T_C)
        theta = self.transport_delay_s(load_percent, P_bar, T_C)
        return tau + theta / 3

# =====================================================
# 3. مدل تشعشع از مشعل‌ها (Radiation Model)
# =====================================================
class BurnerRadiationModel:
    """مدل تابش حرارتی از مشعل‌ها به سوپرهیترها"""
    
    def __init__(self):
        # موقعیت مشعل‌های لایه A (9m ارتفاع)
        self.layer_A_elevation = 9.0
        
        # موقعیت سوپرهیترها
        self.platen_elevation = 32.0  # میانگین 31.88-34.77
        self.final_elevation = 41.0   # میانگین 40-42
        
    def effective_heat_flux_W_m2(self, n_burners: int, 
                                  target: str) -> float:
        """
        شار حرارتی مؤثر به سوپرهیتر
        
        Args:
            n_burners: تعداد مشعل‌های فعال (1-5 از لایه A)
            target: 'platen' یا 'final'
        """
        # توان هر مشعل
        P_burner_MW = 40.0
        P_total_MW = n_burners * P_burner_MW
        
        # فاصله عمودی
        if target == 'platen':
            distance_m = abs(self.platen_elevation - self.layer_A_elevation)
        else:  # final
            distance_m = abs(self.final_elevation - self.layer_A_elevation)
        
        # ضریب دید (View Factor) - تقریبی
        # برای مشعل‌های دور: VF کاهش می‌یابد
        view_factor = min(1.0, 5.0 / (distance_m ** 1.5))
        
        # شار حرارتی موثر
        # q" = (P_total × VF × ε) / A_superheater
        emissivity = 0.8
        A_superheater = 100.0  # تقریبی m²
        
        q_flux = (P_total_MW * 1e6 * view_factor * emissivity) / A_superheater
        
        return q_flux
    
    def heat_distribution_ratio(self, n_burners: int) -> Dict[str, float]:
        """
        نسبت توزیع حرارت بین سوپرهیترها
        در low load با مشعل‌های A:
        - پلاتن دریافت حرارت بیشتری می‌کند (نزدیک‌تر است)
        - فاینال حرارت کمتری دریافت می‌کند
        """
        q_platen = self.effective_heat_flux_W_m2(n_burners, 'platen')
        q_final = self.effective_heat_flux_W_m2(n_burners, 'final')
        
        total = q_platen + q_final
        
        return {
            'platen': q_platen / total,
            'final': q_final / total
        }

# =====================================================
# 4. مدل اسپری دی‌سوپرهیتر
# =====================================================
@dataclass
class SprayDesuperheater:
    """مدل اسپری با دینامیک واقعی"""
    name: str
    spray_water_temp_C: float = 180.0
    valve_time_constant_s: float = 5.0  # تاخیر شیر
    atomization_delay_s: float = 10.0   # تاخیر اتمیزاسیون و اختلاط
    
    def temperature_drop_C(self, 
                          steam_mass_flow_kg_s: float,
                          steam_temp_C: float,
                          steam_pressure_bar: float,
                          spray_flow_percent: float) -> float:
        """
        کاهش دما ناشی از اسپری
        ΔT = (ṁ_spray × cp_spray × ΔT_spray) / (ṁ_steam × cp_steam)
        """
        if spray_flow_percent <= 0:
            return 0.0
        
        # دبی اسپری
        spray_mass_flow = steam_mass_flow_kg_s * (spray_flow_percent / 100)
        
        # خواص
        cp_steam = SteamProperties.specific_heat_J_kgK(
            steam_pressure_bar, steam_temp_C
        )
        cp_water = 4186.0  # J/(kg·K)
        
        # بیلان انرژی
        delta_T = (spray_mass_flow * cp_water * 
                   (steam_temp_C - self.spray_water_temp_C)) / \
                  (steam_mass_flow_kg_s * cp_steam)
        
        return delta_T
    
    def dynamic_response(self, t: float, spray_command: float) -> float:
        """
        پاسخ دینامیکی شیر اسپری (first order + delay)
        G(s) = e^(-θs) / (τs + 1)
        """
        if t < self.atomization_delay_s:
            return 0.0
        
        t_eff = t - self.atomization_delay_s
        response = spray_command * (1 - np.exp(-t_eff / self.valve_time_constant_s))
        
        return response

# =====================================================
# 5. کنترلر PID پیشرفته
# =====================================================
class AdvancedPIDController:
    """کنترلر PID با anti-windup و rate limiting"""
    
    def __init__(self, Kp: float, Ki: float, Kd: float, dt: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        
        # محدودیت‌ها
        self.output_min = 0.0
        self.output_max = 15.0  # حداکثر 15% spray
        self.rate_limit = 2.0   # حداکثر 2% تغییر در هر ثانیه
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
    
    def update(self, setpoint: float, measured: float) -> float:
        error = setpoint - measured
        
        # Proportional
        P = self.Kp * error
        
        # Integral با anti-windup
        self.integral += error * self.dt
        # محدود کردن integral برای جلوگیری از windup
        max_integral = 50.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        I = self.Ki * self.integral
        
        # Derivative با فیلتر
        derivative = (error - self.prev_error) / self.dt
        alpha = 0.1  # فیلتر نویز
        derivative_filtered = alpha * derivative + (1 - alpha) * 0
        D = self.Kd * derivative_filtered
        
        # خروجی کل
        output_raw = P + I + D
        
        # محدودسازی
        output_limited = np.clip(output_raw, self.output_min, self.output_max)
        
        # Rate limiting
        max_change = self.rate_limit * self.dt
        output_change = output_limited - self.prev_output
        output_change = np.clip(output_change, -max_change, max_change)
        output = self.prev_output + output_change
        
        # به‌روزرسانی
        self.prev_error = error
        self.prev_output = output
        
        return output

# =====================================================
# 6. شبیه‌سازی کامل سیستم در Low Load
# =====================================================
class LowLoadBoilerSimulation:
    """شبیه‌سازی کامل بویلر در شرایط راه‌اندازی"""
    
    def __init__(self, load_percent: float = 20.0, n_burners: int = 3):
        self.load_percent = load_percent
        self.n_burners = n_burners
        
        # شرایط عملیاتی
        self.steam_flow_kg_s = (load_percent / 100) * 278.0
        self.pressure_bar = 35.0 + (load_percent / 100) * 65.0
        
        # سوپرهیترها
        self.platen = SuperheaterDynamicModel(
            name="Platen",
            tube_length_m=50.0
        )
        self.final = SuperheaterDynamicModel(
            name="Final",
            tube_length_m=72.0
        )
        
        # مدل تابش
        self.radiation = BurnerRadiationModel()
        
        # اسپری مرحله اول (بعد از پلاتن)
        self.spray1 = SprayDesuperheater(
            name="Spray-1",
            atomization_delay_s=10.0  # تاخیر اصلی
        )
        
        # اسپری مرحله دوم (بعد از فاینال)
        self.spray2 = SprayDesuperheater(
            name="Spray-2",
            atomization_delay_s=10.0
        )
        
        # کنترلرها
        self.pid1 = AdvancedPIDController(
            Kp=0.5, Ki=0.02, Kd=2.0, dt=1.0
        )
        self.pid2 = AdvancedPIDController(
            Kp=0.3, Ki=0.015, Kd=1.5, dt=1.0
        )
    
    def simulate(self, duration_s: float = 1200, dt: float = 1.0) -> Dict:
        """
        شبیه‌سازی کامل
        
        Args:
            duration_s: مدت شبیه‌سازی (پیش‌فرض: 20 دقیقه)
            dt: گام زمانی (ثانیه)
        """
        time = np.arange(0, duration_s, dt)
        n = len(time)
        
        # آرایه‌های ذخیره نتایج
        results = {
            'time_min': time / 60,  # دقیقه
            'temp_platen_in': np.zeros(n),
            'temp_platen_out': np.zeros(n),
            'temp_after_spray1': np.zeros(n),
            'temp_final_out': np.zeros(n),
            'temp_after_spray2': np.zeros(n),
            'spray1_command': np.zeros(n),
            'spray2_command': np.zeros(n),
            'spray1_actual': np.zeros(n),
            'spray2_actual': np.zeros(n),
        }
        
        # شرایط اولیه
        T_furnace_outlet = 950.0  # دمای خروجی کوره
        results['temp_platen_in'][0] = T_furnace_outlet
        results['temp_platen_out'][0] = 420.0
        results['temp_after_spray1'][0] = 400.0
        results['temp_final_out'][0] = 380.0
        results['temp_after_spray2'][0] = 370.0
        
        # تنظیمات (Setpoints)
        SP1 = 410.0  # بعد از spray 1
        SP2 = 400.0  # نهایی
        
        # پارامترهای دینامیکی در این بار
        tau_platen = self.platen.thermal_time_constant_s(
            self.load_percent, self.pressure_bar, 400
        )
        theta_platen = self.platen.transport_delay_s(
            self.load_percent, self.pressure_bar, 400
        )
        
        tau_final = self.final.thermal_time_constant_s(
            self.load_percent, self.pressure_bar, 400
        )
        theta_final = self.final.transport_delay_s(
            self.load_percent, self.pressure_bar, 400
        )
        
        print(f"\n🔧 پارامترهای دینامیکی در بار {self.load_percent}%:")
        print(f"   Platen: τ={tau_platen:.1f}s, θ={theta_platen:.1f}s")
        print(f"   Final:  τ={tau_final:.1f}s, θ={theta_final:.1f}s")
        print(f"   تعداد مشعل‌های فعال: {self.n_burners} (لایه A)")
        
        # شبیه‌سازی
        for i in range(1, n):
            t = time[i]
            
            # 1️⃣ خروجی سوپرهیتر پلاتن (First Order)
            T_in = results['temp_platen_in'][i-1]
            T_out_prev = results['temp_platen_out'][i-1]
            
            dT_dt = (T_in - T_out_prev) / tau_platen
            results['temp_platen_out'][i] = T_out_prev + dT_dt * dt
            
            # 2️⃣ کنترلر اسپری 1
            results['spray1_command'][i] = self.pid1.update(
                SP1, results['temp_after_spray1'][i-1]
            )
            
            # 3️⃣ پاسخ دینامیکی شیر اسپری 1
            results['spray1_actual'][i] = self.spray1.dynamic_response(
                t, results['spray1_command'][i]
            )
            
            # 4️⃣ تأثیر اسپری 1
            delta_T1 = self.spray1.temperature_drop_C(
                self.steam_flow_kg_s,
                results['temp_platen_out'][i],
                self.pressure_bar,
                results['spray1_actual'][i]
            )
            results['temp_after_spray1'][i] = results['temp_platen_out'][i] - delta_T1
            
            # 5️⃣ خروجی سوپرهیتر فاینال
            T_final_in = results['temp_after_spray1'][i]
            T_final_prev = results['temp_final_out'][i-1]
            
            dT_dt_final = (T_final_in - T_final_prev) / tau_final
            results['temp_final_out'][i] = T_final_prev + dT_dt_final * dt
            
            # 6️⃣ کنترلر اسپری 2
            results['spray2_command'][i] = self.pid2.update(
                SP2, results['temp_after_spray2'][i-1]
            )
            
            # 7️⃣ پاسخ دینامیکی شیر اسپری 2
            results['spray2_actual'][i] = self.spray2.dynamic_response(
                t, results['spray2_command'][i]
            )
            
            # 8️⃣ تأثیر اسپری 2
            delta_T2 = self.spray2.temperature_drop_C(
                self.steam_flow_kg_s,
                results['temp_final_out'][i],
                self.pressure_bar,
                results['spray2_actual'][i]
            )
            results['temp_after_spray2'][i] = results['temp_final_out'][i] - delta_T2
            
            # ورودی پلاتن (ثابت در این شبیه‌سازی)
            results['temp_platen_in'][i] = T_furnace_outlet
        
        return results

# =====================================================
# 7. رسم نمودارها
# =====================================================
def plot_results(results: Dict, load_percent: float, n_burners: int):
    """رسم نمودارهای کامل"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # نمودار 1: دماها
    ax1 = axes[0]
    ax1.plot(results['time_min'], results['temp_platen_out'], 
             'b-', linewidth=2, label='خروجی پلاتن')
    ax1.plot(results['time_min'], results['temp_after_spray1'], 
             'g-', linewidth=2, label='بعد از اسپری 1')
    ax1.plot(results['time_min'], results['temp_final_out'], 
             'r-', linewidth=2, label='خروجی فاینال')
    ax1.plot(results['time_min'], results['temp_after_spray2'], 
             'm-', linewidth=2.5, label='خروجی نهایی (بعد اسپری 2)')
    ax1.axhline(410, color='g', linestyle='--', alpha=0.5, label='SP1=410°C')
    ax1.axhline(400, color='m', linestyle='--', alpha=0.5, label='SP2=400°C')
    ax1.axvline(10, color='gray', linestyle=':', alpha=0.7, label='تاخیر اسپری (10 دقیقه)')
    ax1.set_ylabel('دما (°C)', fontsize=12, fontweight='bold')
    ax1.set_title(f'شبیه‌سازی دینامیکی بویلر - بار {load_percent}% - {n_burners} مشعل لایه A', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # نمودار 2: دبی اسپری 1
    ax2 = axes[1]
    ax2.plot(results['time_min'], results['spray1_command'], 
             'b--', linewidth=1.5, label='فرمان اسپری 1', alpha=0.7)
    ax2.plot(results['time_min'], results['spray1_actual'], 
             'b-', linewidth=2, label='اسپری واقعی 1')
    ax2.axvline(10, color='gray', linestyle=':', alpha=0.7)
    ax2.set_ylabel('دبی اسپری 1 (%)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # نمودار 3: دبی اسپری 2
    ax3 = axes[2]
    ax3.plot(results['time_min'], results['spray2_command'], 
             'r--', linewidth=1.5, label='فرمان اسپری 2', alpha=0.7)
    ax3.plot(results['time_min'], results['spray2_actual'], 
             'r-', linewidth=2, label='اسپری واقعی 2')
    ax3.axvline(10, color='gray', linestyle=':', alpha=0.7)
    ax3.set_xlabel('زمان (دقیقه)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('دبی اسپری 2 (%)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ذخیره
    filename = f'low_load_{int(load_percent)}percent_{n_burners}burners.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ نمودار ذخیره شد: {filename}")
    
    return fig

# =====================================================
# 8. اجرای شبیه‌سازی
# =====================================================
if __name__ == "__main__":
    print("="*60)
    print("🚀 شبیه‌سازی دینامیکی اسپری در شرایط Low Load")
    print("   بویلر: 1000 TPH, 170 bar, 540°C")
    print("="*60)
    
    # شبیه‌سازی برای بارهای مختلف
    test_cases = [
        {'load': 10, 'burners': 1},
        {'load': 20, 'burners': 3},
        {'load': 30, 'burners': 5},
    ]
    
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"📊 سناریو: بار {case['load']}% - {case['burners']} مشعل")
        print(f"{'='*60}")
        
        # ایجاد شبیه‌سازی
        sim = LowLoadBoilerSimulation(
            load_percent=case['load'],
            n_burners=case['burners']
        )
        
        # اجرا
        results = sim.simulate(duration_s=1200, dt=1.0)
        
        # رسم نمودار
        plot_results(results, case['load'], case['burners'])
        
        # گزارش نتایج
        print(f"\n📈 نتایج نهایی (t=20 min):")
        print(f"   دمای خروجی نهایی: {results['temp_after_spray