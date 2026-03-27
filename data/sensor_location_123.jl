#A相缺相：3、4、5、6、7、13、16、17、18、23、25、32、33、35、39、40、42、44、59、60、74、75、76、85、86、91、93、97、103、104、105、107、108  ##33
#B相缺相：4、5、6、7、10、11、12、15、16、17、18、20、21、25、27、28、32、33、34、35、38、42、46、47、69、70、71、72、74、75、76、85、86、89、93、95、103、104、105、110、111、112、113、114、115、128、131、132  ##48
#C相缺相：3、10、11、12、13、15、20、21、23、34、37、38、39、40、44、46、47、59、60、69、70、71、72、89、91、95、97、107、108、110、111、112、113、114、115、131、132   ##37




function FAD10_config_ieee123()   #138 
    pmu_buses=[2, 8, 22, 36, 43, 53] # 54
    scada_buses=[7, 18, 28, 38, 48, 58, 68, 98,118]# 20
    ami_buses=[4, 7, 12,17,27,33,37,40,47,50,55,60,67,69,70,77,89,97,107,119]#64
    pmu_phases=Dict(2=>["a","b","c"], 8=>["a","b","c"],22=>["a","b","c"],36=>["a","b","c"],43=>["a","b","c"],53=>["a","b","c"])  # 指定每个PMU节点要提取的相位

    scada_phases=Dict(7=>["c"], 18=>["c"], 28=>["a","c"], 38=>["a"], 48=>["a","b","c"], 58=>["a","b","c"], 68=>["a","b","c"],98=>["a","b","c"],118=>["a","b","c"])

    ami_phases=Dict(4=>["c"], 7=>["c"], 12=>["a"],17=>["c"], 27=>["a","c"], 
    33=>["c"], 37=>["a","b"], 40=>["b"], 47=>["a"], 50=>["a","b","c"], 55=>["a","b","c"], 60=>["b"], 67=>["a","b","c"], 69=>["a"], 70=>["a"], 
    77=>["a","b","c"], 89=>["a"], 97=>["b"], 107=>["b"],119=>["a","b","c"]
    )

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end

function get_phases(bus_id, phase_dict)
    return get(phase_dict, bus_id, ["a", "b", "c"])
end

function FAD20_config_ieee123()   #276
    pmu_buses=[2, 8, 22, 36, 43, 53, 61, 77, 90] # 81
    scada_buses=[7, 18, 28, 38, 48, 58, 68, 78, 88, 98, 118, 126]# 29
    ami_buses=[4, 7, 9,12, 15,17,19,20,22,25,27,30,33,34,37,40,42,47,50,55,57,60,63,67,69,70,73,77,80,82,85,89,90,97,100,107,110,113,115,117,119,123,129]#166
    pmu_phases=Dict(2=>["a","b","c"], 8=>["a","b","c"],22=>["a","b","c"],36=>["a","b","c"],43=>["a","b","c"],53=>["a","b","c"],61=>["a","b","c"], 77=>["a","b","c"], 90=>["a","b","c"])  # 指定每个PMU节点要提取的相位

    scada_phases=Dict(7=>["c"], 18=>["c"], 28=>["a","c"], 38=>["a"], 48=>["a","b","c"], 58=>["a","b","c"], 68=>["a","b","c"], 78=>["a","b","c"], 88=>["a","b","c"], 98=>["a","b","c"],118=>["a","b","c"],
     126=>["a","b","c"])

    ami_phases=Dict(4=>["c"], 7=>["c"], 9=>["a","b","c"], 12=>["a"], 15=>["a"], 17=>["c"],19=>["a","b","c"],20=>["a"], 22=>["a","b","c"], 25=>["c"], 27=>["a","c"], 30=>["a","b","c"],
    33=>["c"], 34=>["a"], 37=>["a","b"], 40=>["b"], 42=>["c"], 47=>["a"], 50=>["a","b","c"], 55=>["a","b","c"], 57=>["a","b","c"], 60=>["b"], 63=>["a","b","c"], 67=>["a","b","c"], 69=>["a"], 70=>["a"], 
    73=>["a","b","c"], 77=>["a","b","c"], 80=>["a","b","c"], 82=>["a","b","c"], 85=>["c"], 89=>["a"], 90=>["a","b","c"], 97=>["b"],100=>["a","b","c"], 107=>["b"],110=>["a"], 113=>["a"], 115=>["a"],117=>["a","b","c"],119=>["a","b","c"],
    123=>["a","b","c"], 129=>["a","b","c"]
    )

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end




function get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    # --- 生成 PMU Sensors ---
    # 逻辑：按 测量类型 分组 (先全是Vmag，再全是Real...)
    pmu_sensors = Tuple{String, String, Symbol}[]
    for bus in pmu_buses
        phases = get_phases(bus, pmu_phases)
        # PMU 的顺序逻辑：先遍历测量类型，再遍历相位
        for meas_type in [:Vmag, :V_real, :V_imag]
            for p in phases
                push!(pmu_sensors, (string(bus), p, meas_type))
            end
        end
    end

    # --- 生成 SCADA Sensors ---
    # 逻辑：只有 Vmag
    scada_sensors = Tuple{String, String, Symbol}[]
    for bus in scada_buses
        phases = get_phases(bus, scada_phases)
        for p in phases
            push!(scada_sensors, (string(bus), p, :Vmag))
        end
    end

    # --- 生成 AMI Sensors ---
    # 逻辑：按 相位 分组 (A相的P/Q，然后B相的P/Q...)
    ami_sensors = Tuple{String, String, Symbol}[]
    for bus in ami_buses
        phases = get_phases(bus, ami_phases)
        # AMI 的顺序逻辑：先遍历相位，再遍历测量类型(P, Q)
        for p in phases
            for meas_type in [:P_kW, :Q_kVAR]
                push!(ami_sensors, (string(bus), p, meas_type))
            end
        end
    end

    return pmu_sensors, scada_sensors, ami_sensors
end