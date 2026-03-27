function FAD10_config()    
    pmu_buses=[3]
    scada_buses=[8,12,15]
    ami_buses=[7,13,18,22,25,29,30,33]
    pmu_phases=Dict(3=>["a","b","c"])  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(8=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"])
    ami_phases=Dict(7=>["b","c"], 13=>["a","c"], 18=>["a","b"], 22=>["b"], 25=>["a","c"], 29=>["a","b"], 30=>["b","c"], 33=>["a","c"])

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end

function get_phases(bus_id, phase_dict)
    return get(phase_dict, bus_id, ["a", "b", "c"])
end

function FAD20_config()    
    pmu_buses=[3]
    scada_buses=[8,12,15,19,24,27,31]
    ami_buses=[7,9,13,15,18,19,22,25,29,30,33]
    pmu_phases=Dict(3=>["a","b","c"])  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(8=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"], 19=>["a","b","c"], 24=>["a","b","c"],27=>["a","b","c"],31=>["a","b","c"])
    ami_phases=Dict(7=>["a","b","c"], 9=>["a","b","c"], 13=>["a","b","c"], 15=>["a","b","c"], 18=>["a","b","c"], 19=>["a","b","c"], 22=>["a","b","c"], 25=>["a","b","c"], 29=>["a","b","c"], 
    30=>["a","b","c"], 33=>["a","b","c"]
    )

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end

function FAD30_config()    
    pmu_buses=[3,18,22,29]
    scada_buses=[5,8,12,15,19,24,27,31]
    ami_buses=[2,4,7,9,13,15,18,19,22,25,27,29,30,33]
    pmu_phases=Dict(3=>["a","b","c"],18=>["a","b","c"],22=>["a","b","c"],29=>["a","b","c"])  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(5=>["a","b","c"], 8=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"], 19=>["a","b","c"], 24=>["a","b","c"],27=>["a","b","c"],31=>["a","b","c"])
    ami_phases=Dict(2=>["a","b","c"], 4=>["a","b","c"], 7=>["a","b","c"], 9=>["a","b","c"], 13=>["a","b","c"], 15=>["a","b","c"], 18=>["a","b","c"], 19=>["a","b","c"], 22=>["a","b","c"], 25=>["a","b","c"], 27=>["a","b","c"], 
    29=>["a","b","c"], 30=>["a","b","c"], 33=>["a","b","c"]
    )

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end

function FAD40_config()    
    pmu_buses=[3,6,18,22,29,32]
    scada_buses=[5,8,10,12,15,17,19,21,24,27,31,33]
    ami_buses=[2,3,4,6,7,9,13,15,18,19,22,24,25,27,29,30,33]
    pmu_phases=Dict(3=>["a","b","c"],6=>["a","b","c"],18=>["a","b","c"],22=>["a","b","c"],29=>["a","b","c"],32=>["a","b","c"])  # 指定每个PMU节点要提取的相位
    scada_phases=Dict( 5=>["a","b","c"], 8=>["a","b","c"], 10=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"], 17=>["a","b","c"], 19=>["a","b","c"], 21=>["a","b","c"],24=>["a","b","c"],27=>["a","b","c"],
    31=>["a","b","c"],33=>["a","b","c"])
    ami_phases=Dict(2=>["a","b","c"], 3=>["a","b","c"], 4=>["a","b","c"], 6=>["a","b","c"], 7=>["a","b","c"], 9=>["a","b","c"], 13=>["a","b","c"], 15=>["a","b","c"], 18=>["a","b","c"], 19=>["a","b","c"], 22=>["a","b","c"], 24=>["a","b","c"], 
    25=>["a","b","c"], 27=>["a","b","c"], 29=>["a","b","c"], 30=>["a","b","c"], 33=>["a","b","c"]
    )

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end

function FAD50_config()    
    pmu_buses=[3,4,6,13,18,22,25,29,32]
    scada_buses=[2,5,8,10,12,15,17,19,21,24,27,31,33]
    ami_buses=[2,3,4,5,6,7,9,11,13,15,16,18,19,22,24,25,27,29,30,33]
    pmu_phases=Dict(3=>["a","b","c"],4=>["a","b","c"],6=>["a","b","c"],13=>["a","b","c"],18=>["a","b","c"],22=>["a","b","c"],25=>["a","b","c"],29=>["a","b","c"],32=>["a","b","c"])  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(2=>["a","b","c"], 5=>["a","b","c"], 8=>["a","b","c"], 10=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"], 17=>["a","b","c"], 19=>["a","b","c"], 21=>["a","b","c"],24=>["a","b","c"],27=>["a","b","c"],
    31=>["a","b","c"],33=>["a","b","c"])
    ami_phases=Dict(2=>["a","b","c"], 3=>["a","b","c"], 4=>["a","b","c"],5=>["a","b","c"], 6=>["a","b","c"], 7=>["a","b","c"], 9=>["a","b","c"], 11=>["a","b","c"], 13=>["a","b","c"], 15=>["a","b","c"], 16=>["a","b","c"], 18=>["a","b","c"], 19=>["a","b","c"], 22=>["a","b","c"], 24=>["a","b","c"], 
    25=>["a","b","c"], 27=>["a","b","c"], 29=>["a","b","c"], 30=>["a","b","c"], 33=>["a","b","c"]
    )

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end

function FAD60_config()    
    pmu_buses=[3,4,6,9,13,18,22,23,25,29,32]
    scada_buses=[2,5,7,8,10,12,15,17,19,21,24,27,30,31,33]
    ami_buses=[2,3,4,5,6,7,8,9,11,13,15,16,17,18,19,22,24,25,26,27,29,30,31,33]
    pmu_phases=Dict(3=>["a","b","c"],4=>["a","b","c"],6=>["a","b","c"],9=>["a","b","c"],13=>["a","b","c"],18=>["a","b","c"],22=>["a","b","c"],23=>["a","b","c"],25=>["a","b","c"],29=>["a","b","c"],32=>["a","b","c"])  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(2=>["a","b","c"], 5=>["a","b","c"], 7=>["a","b","c"], 8=>["a","b","c"], 10=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"], 17=>["a","b","c"], 19=>["a","b","c"], 21=>["a","b","c"],24=>["a","b","c"],27=>["a","b","c"],
    30=>["a","b","c"],31=>["a","b","c"],33=>["a","b","c"])
    ami_phases=Dict(2=>["a","b","c"], 3=>["a","b","c"], 4=>["a","b","c"],5=>["a","b","c"], 6=>["a","b","c"], 7=>["a","b","c"], 8=>["a","b","c"], 9=>["a","b","c"], 11=>["a","b","c"], 
    13=>["a","b","c"], 15=>["a","b","c"], 16=>["a","b","c"], 17=>["a","b","c"], 18=>["a","b","c"], 19=>["a","b","c"], 22=>["a","b","c"], 24=>["a","b","c"],  25=>["a","b","c"], 26=>["a","b","c"], 
    27=>["a","b","c"], 29=>["a","b","c"], 30=>["a","b","c"], 31=>["a","b","c"],33=>["a","b","c"]
    )

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end

function FAD70_config()    
    pmu_buses=[3,4,6,9,11,13,14,18,22,23,25,29,32]
    scada_buses=[2,5,7,8,10,12,15,16,17,19,20,21,24,27,30,31,33]
    ami_buses=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,25,26,27,29,30,31,33]
    pmu_phases=Dict(3=>["a","b","c"],4=>["a","b","c"],6=>["a","b","c"],9=>["a","b","c"],11=>["a","b","c"],13=>["a","b","c"],14=>["a","b","c"],18=>["a","b","c"],22=>["a","b","c"],23=>["a","b","c"],25=>["a","b","c"],29=>["a","b","c"],32=>["a","b","c"])  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(2=>["a","b","c"], 5=>["a","b","c"], 7=>["a","b","c"], 8=>["a","b","c"], 10=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"], 16=>["a","b","c"], 17=>["a","b","c"], 19=>["a","b","c"], 20=>["a","b","c"], 21=>["a","b","c"],24=>["a","b","c"],27=>["a","b","c"],
    30=>["a","b","c"],31=>["a","b","c"],33=>["a","b","c"])
    ami_phases=Dict(2=>["a","b","c"], 3=>["a","b","c"], 4=>["a","b","c"],5=>["a","b","c"], 6=>["a","b","c"], 7=>["a","b","c"], 8=>["a","b","c"], 9=>["a","b","c"], 10=>["a","b","c"], 11=>["a","b","c"], 12=>["a","b","c"],
    13=>["a","b","c"], 14=>["a","b","c"], 15=>["a","b","c"], 16=>["a","b","c"], 17=>["a","b","c"], 18=>["a","b","c"], 19=>["a","b","c"], 20=>["a","b","c"], 22=>["a","b","c"], 24=>["a","b","c"],  25=>["a","b","c"], 26=>["a","b","c"], 
    27=>["a","b","c"], 29=>["a","b","c"], 30=>["a","b","c"], 31=>["a","b","c"],33=>["a","b","c"]
    )

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end

function FAD80_config()    
    pmu_buses=[3,4,6,9,11,13,14,16,18,22,23,25,26,28,29,32]
    scada_buses=[2,5,7,8,10,12,15,17,19,20,21,24,27,30,31,33]
    ami_buses=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    pmu_phases=Dict(3=>["a","b","c"],4=>["a","b","c"],6=>["a","b","c"],9=>["a","b","c"],11=>["a","b","c"],13=>["a","b","c"],14=>["a","b","c"],16=>["a","b","c"],18=>["a","b","c"],22=>["a","b","c"],23=>["a","b","c"],25=>["a","b","c"],26=>["a","b","c"],28=>["a","b","c"],29=>["a","b","c"],32=>["a","b","c"])  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(2=>["a","b","c"], 5=>["a","b","c"], 7=>["a","b","c"], 8=>["a","b","c"], 10=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"], 17=>["a","b","c"], 19=>["a","b","c"], 20=>["a","b","c"], 21=>["a","b","c"],24=>["a","b","c"],27=>["a","b","c"],
    30=>["a","b","c"],31=>["a","b","c"],33=>["a","b","c"])
    ami_phases=Dict(2=>["a","b","c"], 3=>["a","b","c"], 4=>["a","b","c"],5=>["a","b","c"], 6=>["a","b","c"], 7=>["a","b","c"], 8=>["a","b","c"], 9=>["a","b","c"], 10=>["a","b","c"], 11=>["a","b","c"], 12=>["a","b","c"],
    13=>["a","b","c"], 14=>["a","b","c"], 15=>["a","b","c"], 16=>["a","b","c"], 17=>["a","b","c"], 18=>["a","b","c"], 19=>["a","b","c"], 20=>["a","b","c"], 21=>["a","b","c"], 22=>["a","b","c"], 23=>["a","b","c"], 24=>["a","b","c"],
    25=>["a","b","c"], 26=>["a","b","c"], 27=>["a","b","c"], 28=>["a","b","c"], 29=>["a","b","c"], 30=>["a","b","c"], 31=>["a","b","c"],32=>["a","b","c"], 33=>["a","b","c"]
    )

    pmu_sensors, scada_sensors, ami_sensors = get_sensor_informations(pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases)
    return pmu_buses, scada_buses, ami_buses, pmu_phases, scada_phases, ami_phases, pmu_sensors, scada_sensors, ami_sensors
end

function FAD90_config()    
    pmu_buses=[3,4,5,6,8,9,11,12,13,14,16,17,18,20,21,22,23,25,26,28,29,30,31,32]
    scada_buses=[2,7,10,15,19,24,27,33]
    ami_buses=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    pmu_phases=Dict(3=>["a","b","c"],4=>["a","b","c"],5=>["a","b","c"],6=>["a","b","c"],8=>["a","b","c"],9=>["a","b","c"],11=>["a","b","c"],12=>["a","b","c"],13=>["a","b","c"],14=>["a","b","c"],16=>["a","b","c"],17=>["a","b","c"],18=>["a","b","c"],20=>["a","b","c"],21=>["a","b","c"],22=>["a","b","c"],23=>["a","b","c"],25=>["a","b","c"],26=>["a","b","c"],28=>["a","b","c"],29=>["a","b","c"],30=>["a","b","c"],31=>["a","b","c"],32=>["a","b","c"])  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(2=>["a","b","c"], 7=>["a","b","c"], 10=>["a","b","c"], 15=>["a","b","c"], 19=>["a","b","c"], 24=>["a","b","c"],27=>["a","b","c"],33=>["a","b","c"])
    ami_phases=Dict(2=>["a","b","c"], 3=>["a","b","c"], 4=>["a","b","c"],5=>["a","b","c"], 6=>["a","b","c"], 7=>["a","b","c"], 8=>["a","b","c"], 9=>["a","b","c"], 10=>["a","b","c"], 11=>["a","b","c"], 12=>["a","b","c"],
    13=>["a","b","c"], 14=>["a","b","c"], 15=>["a","b","c"], 16=>["a","b","c"], 17=>["a","b","c"], 18=>["a","b","c"], 19=>["a","b","c"], 20=>["a","b","c"], 21=>["a","b","c"], 22=>["a","b","c"], 23=>["a","b","c"], 24=>["a","b","c"],
    25=>["a","b","c"], 26=>["a","b","c"], 27=>["a","b","c"], 28=>["a","b","c"], 29=>["a","b","c"], 30=>["a","b","c"], 31=>["a","b","c"],32=>["a","b","c"], 33=>["a","b","c"]
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