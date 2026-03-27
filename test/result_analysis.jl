voltage_mag_a = result_bmc.X_mean[1:3:end,5]
voltage_actual_mag_a = Vmag_abc[:,1]
voltage_mape = sum(abs.(voltage_mag_a - voltage_actual_mag_a[2:end]) ./ voltage_actual_mag_a[2:end])/32