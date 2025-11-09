from steamlib.burners import BurnerSystem

b = BurnerSystem()
print('✅ Burner system loaded!')
print(f'  Total: {len(b)} burners, {b.total_thermal_power_MW} MW')
print(f'  Layer B elevation: {b.get_burners_by_layer("B")[0].elevation_m} m')