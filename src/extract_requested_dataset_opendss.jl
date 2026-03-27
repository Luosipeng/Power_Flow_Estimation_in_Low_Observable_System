function extract_requested_dataset_opendss(voltage_mag_a, voltage_mag_b, voltage_mag_c,
            voltage_ang_a, voltage_ang_b, voltage_ang_c,
            power_p_a, power_p_b, power_p_c,
            power_q_a, power_q_b, power_q_c;
    pmu_buses=[3], 
    scada_buses=[8,12,15,32], 
    ami_buses=[7,13,16,18,22,25,29,30,33],
    pmu_phases=Dict(3=>["a","b","c"]),  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(8=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"], 32=>["a","b","c"]),
    ami_phases=Dict(7=>["b","c"], 13=>["a","c"], 16=>["a","b"], 18=>["a","b"], 22=>["b","c"], 25=>["a","c"], 29=>["a","b"], 30=>["b","c"], 33=>["a","c"])
)    
   
    # # Assian Voltage values
    # Vmag_out_a = voltage_mag_a[2:end, :]
    # Vmag_out_b = voltage_mag_b[2:end, :]
    # Vmag_out_c = voltage_mag_c[2:end, :]
    # Vang_out_a = voltage_ang_a[2:end, :]
    # Vang_out_b = voltage_ang_b[2:end, :]
    # Vang_out_c = voltage_ang_c[2:end, :]
    Vmag_out_a = voltage_mag_a
    Vmag_out_b = voltage_mag_b
    Vmag_out_c = voltage_mag_c
    Vang_out_a = voltage_ang_a
    Vang_out_b = voltage_ang_b
    Vang_out_c = voltage_ang_c

    # # Assian Power values
    # Pd_out_a = power_p_a[2:end, :]
    # Pd_out_b = power_p_b[2:end, :]
    # Pd_out_c = power_p_c[2:end, :]
    # Qd_out_a = power_q_a[2:end, :]
    # Qd_out_b = power_q_b[2:end, :]
    # Qd_out_c = power_q_c[2:end, :]
    Pd_out_a = power_p_a
    Pd_out_b = power_p_b
    Pd_out_c = power_p_c
    Qd_out_a = power_q_a
    Qd_out_b = power_q_b
    Qd_out_c = power_q_c


    # 提取所需的测量值
    pmu_dict = extract_pmu_data(Vmag_out_a, Vmag_out_b, Vmag_out_c, Vang_out_a, Vang_out_b, Vang_out_c, pmu_buses, pmu_phases)
    scada_dict = extract_scada_data(Vmag_out_a, Vmag_out_b, Vmag_out_c, scada_buses, scada_phases)
    ami_dict = extract_ami_data(Pd_out_a,Pd_out_b,Pd_out_c, Qd_out_a, Qd_out_b, Qd_out_c, ami_buses, ami_phases)

    return Dict(
        :PMU => pmu_dict,
        :SCADA => scada_dict,
        :AMI => ami_dict
    )
end

function extract_pmu_data(Vmag_out_a, Vmag_out_b, Vmag_out_c, Vang_out_a, Vang_out_b, Vang_out_c, pmu_buses, pmu_phases)
    pmu_data = Dict{String, Dict{String, Dict{Symbol, Any}}}()
    
    for bus in pmu_buses
        bus_str = string(bus)
        phases = haskey(pmu_phases, bus) ? pmu_phases[bus] : ["a", "b", "c"]
        
        pmu_data[bus_str] = Dict{String, Dict{Symbol, Any}}()
        
        for phase in phases
            # 获取该节点该相位的行索引
            if phase == "a"
                Vmag = Vmag_out_a[bus, :]             # 1×T 或向量
                Vang_deg = Vang_out_a[bus, :]         # 1×T 或向量(度)
                
            elseif phase == "b"
                Vmag = Vmag_out_b[bus, :]             # 1×T 或向量
                Vang_deg = Vang_out_b[bus, :]         # 1×T 或向量(度)

            elseif phase == "c"
                Vmag = Vmag_out_c[bus, :]             # 1×T 或向量
                Vang_deg = Vang_out_c[bus, :]         # 1×T 或向量(度)
        
            end
            θ = deg2rad.(Vang_deg)                    # 弧度
            V_real = Vmag .* cos.(θ)
            V_imag = Vmag .* sin.(θ)

            pmu_data[bus_str][phase] = Dict(
                :Times => collect(1:size(Vmag, 1)),  # 假设时间步长为1
                :Vmag => Vmag,
                :V_real => V_real,
                :V_imag => V_imag,
            )
        end
    end
    return pmu_data
end


function extract_scada_data(Vmag_out_a, Vmag_out_b, Vmag_out_c, scada_buses, scada_phases)
    scada_data = Dict{String, Dict{String, Dict{Symbol, Any}}}()
    
     for bus in scada_buses
        bus_str = string(bus)
        phases = haskey(scada_phases, bus) ? scada_phases[bus] : ["a", "b", "c"]
        
        scada_data[bus_str] = Dict{String, Dict{Symbol, Any}}()
        
        for phase in phases
            # 获取该节点该相位的行索引
            if phase == "a"
                Vmag = Vmag_out_a[bus, :]             # 1×T 或向量
                
            elseif phase == "b"
                Vmag = Vmag_out_b[bus, :]             # 1×T 或向量

            elseif phase == "c"
                Vmag = Vmag_out_c[bus, :]             # 1×T 或向量
        
            end

            scada_data[bus_str][phase] = Dict(
                :Times => collect(1:600:size(Vmag, 1)),  # 假设时间步长为1
                :Vmag => Vmag[1:600:end],
            )
        end
    end
    return scada_data
end

function extract_ami_data(Pd_out_a, Pd_out_b, Pd_out_c, Qd_out_a, Qd_out_b, Qd_out_c, ami_buses, ami_phases)
    ami_data = Dict{String, Dict{String, Dict{Symbol, Any}}}()
    
    for bus in ami_buses
        bus_str = string(bus)
        phases = haskey(ami_phases, bus) ? ami_phases[bus] : ["a", "b", "c"]
        
        ami_data[bus_str] = Dict{String, Dict{Symbol, Any}}()
        
        for phase in phases
            # 获取该节点该相位的行索引
            if phase == "a"
                P_net = Pd_out_a[bus, :]             # 1×T 或向量
                Q_net = Qd_out_a[bus, :]             # 1×T 或向量
                
            elseif phase == "b"
                P_net = Pd_out_b[bus, :]             # 1×T 或向量
                Q_net = Qd_out_b[bus, :]             # 1×T 或向量

            elseif phase == "c"
                P_net = Pd_out_c[bus, :]             # 1×T 或向量
                Q_net = Qd_out_c[bus, :]             # 1×T 或向量
        
            end

            ami_data[bus_str][phase] = Dict(
                :Times => collect(1:9000:size(P_net, 1)),  # 假设时间步长为1
                :P_kW  => P_net[1:9000:end],
                :Q_kVAR => Q_net[1:9000:end],
            )
        end
    end
    return ami_data
end