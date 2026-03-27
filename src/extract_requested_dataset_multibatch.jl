function extract_requested_dataset_multibatch(batch_data;
    pmu_buses=[3], 
    scada_buses=[8,12,15,32], 
    ami_buses=[7,13,16,18,22,25,29,30,33],
    pmu_phases=Dict(3=>["a","b","c"]),  # 指定每个PMU节点要提取的相位
    scada_phases=Dict(8=>["a","b","c"], 12=>["a","b","c"], 15=>["a","b","c"], 32=>["a","b","c"]),
    ami_phases=Dict(7=>["b","c"], 13=>["a","c"], 16=>["a","b"], 18=>["a","b"], 22=>["b","c"], 25=>["a","c"], 29=>["a","b"], 30=>["b","c"], 33=>["a","c"])
)    
    # 拼接数据
    batch_data_1 = batch_data[1];   batch_data_2 = batch_data[2];
    batch_data_3 = batch_data[3];   batch_data_4 = batch_data[4];
    batch_data_5 = batch_data[5];   batch_data_6 = batch_data[6];
    batch_data_7 = batch_data[7];   batch_data_8 = batch_data[8];
    batch_data_9 = batch_data[9];   batch_data_10 = batch_data[10];
    batch_data_11 = batch_data[11]; batch_data_12 = batch_data[12];
    batch_data_13 = batch_data[13]; batch_data_14 = batch_data[14];
    batch_data_15 = batch_data[15]; batch_data_16 = batch_data[16];
    batch_data_17 = batch_data[17]; batch_data_18 = batch_data[18];
    
    Vmag_out_a = hcat(batch_data_1.Vmag_out_abc[:,1,:], batch_data_2.Vmag_out_abc[:,1,:],
                    batch_data_3.Vmag_out_abc[:,1,:], batch_data_4.Vmag_out_abc[:,1,:],
                    batch_data_5.Vmag_out_abc[:,1,:], batch_data_6.Vmag_out_abc[:,1,:],
                    batch_data_7.Vmag_out_abc[:,1,:], batch_data_8.Vmag_out_abc[:,1,:],
                    batch_data_9.Vmag_out_abc[:,1,:], batch_data_10.Vmag_out_abc[:,1,:],
                    batch_data_11.Vmag_out_abc[:,1,:], batch_data_12.Vmag_out_abc[:,1,:],
                    batch_data_13.Vmag_out_abc[:,1,:], batch_data_14.Vmag_out_abc[:,1,:],
                    batch_data_15.Vmag_out_abc[:,1,:], batch_data_16.Vmag_out_abc[:,1,:],
                    batch_data_17.Vmag_out_abc[:,1,:], batch_data_18.Vmag_out_abc[:,1,:])

    Vmag_out_b = hcat(batch_data_1.Vmag_out_abc[:,2,:], batch_data_2.Vmag_out_abc[:,2,:],
                    batch_data_3.Vmag_out_abc[:,2,:], batch_data_4.Vmag_out_abc[:,2,:],
                    batch_data_5.Vmag_out_abc[:,2,:], batch_data_6.Vmag_out_abc[:,2,:],
                    batch_data_7.Vmag_out_abc[:,2,:], batch_data_8.Vmag_out_abc[:,2,:],
                    batch_data_9.Vmag_out_abc[:,2,:], batch_data_10.Vmag_out_abc[:,2,:],
                    batch_data_11.Vmag_out_abc[:,2,:], batch_data_12.Vmag_out_abc[:,2,:],
                    batch_data_13.Vmag_out_abc[:,2,:], batch_data_14.Vmag_out_abc[:,2,:],
                    batch_data_15.Vmag_out_abc[:,2,:], batch_data_16.Vmag_out_abc[:,2,:],
                    batch_data_17.Vmag_out_abc[:,2,:], batch_data_18.Vmag_out_abc[:,2,:])

    Vmag_out_c = hcat(batch_data_1.Vmag_out_abc[:,3,:], batch_data_2.Vmag_out_abc[:,3,:],
                    batch_data_3.Vmag_out_abc[:,3,:], batch_data_4.Vmag_out_abc[:,3,:],
                    batch_data_5.Vmag_out_abc[:,3,:], batch_data_6.Vmag_out_abc[:,3,:],
                    batch_data_7.Vmag_out_abc[:,3,:], batch_data_8.Vmag_out_abc[:,3,:],
                    batch_data_9.Vmag_out_abc[:,3,:], batch_data_10.Vmag_out_abc[:,3,:],
                    batch_data_11.Vmag_out_abc[:,3,:], batch_data_12.Vmag_out_abc[:,3,:],
                    batch_data_13.Vmag_out_abc[:,3,:], batch_data_14.Vmag_out_abc[:,3,:],
                    batch_data_15.Vmag_out_abc[:,3,:], batch_data_16.Vmag_out_abc[:,3,:],
                    batch_data_17.Vmag_out_abc[:,3,:], batch_data_18.Vmag_out_abc[:,3,:])


    Vang_out_a = hcat(batch_data_1.Vang_out_abc[:,1,:], batch_data_2.Vang_out_abc[:,1,:],
                    batch_data_3.Vang_out_abc[:,1,:], batch_data_4.Vang_out_abc[:,1,:],
                    batch_data_5.Vang_out_abc[:,1,:], batch_data_6.Vang_out_abc[:,1,:],
                    batch_data_7.Vang_out_abc[:,1,:], batch_data_8.Vang_out_abc[:,1,:],
                    batch_data_9.Vang_out_abc[:,1,:], batch_data_10.Vang_out_abc[:,1,:],
                    batch_data_11.Vang_out_abc[:,1,:], batch_data_12.Vang_out_abc[:,1,:],
                    batch_data_13.Vang_out_abc[:,1,:], batch_data_14.Vang_out_abc[:,1,:],
                    batch_data_15.Vang_out_abc[:,1,:], batch_data_16.Vang_out_abc[:,1,:],
                    batch_data_17.Vang_out_abc[:,1,:], batch_data_18.Vang_out_abc[:,1,:])

    Vang_out_b = hcat(batch_data_1.Vang_out_abc[:,2,:], batch_data_2.Vang_out_abc[:,2,:],
                    batch_data_3.Vang_out_abc[:,2,:], batch_data_4.Vang_out_abc[:,2,:],
                    batch_data_5.Vang_out_abc[:,2,:], batch_data_6.Vang_out_abc[:,2,:],
                    batch_data_7.Vang_out_abc[:,2,:], batch_data_8.Vang_out_abc[:,2,:],
                    batch_data_9.Vang_out_abc[:,2,:], batch_data_10.Vang_out_abc[:,2,:],
                    batch_data_11.Vang_out_abc[:,2,:], batch_data_12.Vang_out_abc[:,2,:],
                    batch_data_13.Vang_out_abc[:,2,:], batch_data_14.Vang_out_abc[:,2,:],
                    batch_data_15.Vang_out_abc[:,2,:], batch_data_16.Vang_out_abc[:,2,:],
                    batch_data_17.Vang_out_abc[:,2,:], batch_data_18.Vang_out_abc[:,2,:])

    Vang_out_c = hcat(batch_data_1.Vang_out_abc[:,3,:], batch_data_2.Vang_out_abc[:,3,:],
                    batch_data_3.Vang_out_abc[:,3,:], batch_data_4.Vang_out_abc[:,3,:],
                    batch_data_5.Vang_out_abc[:,3,:], batch_data_6.Vang_out_abc[:,3,:],
                    batch_data_7.Vang_out_abc[:,3,:], batch_data_8.Vang_out_abc[:,3,:],
                    batch_data_9.Vang_out_abc[:,3,:], batch_data_10.Vang_out_abc[:,3,:],
                    batch_data_11.Vang_out_abc[:,3,:], batch_data_12.Vang_out_abc[:,3,:],
                    batch_data_13.Vang_out_abc[:,3,:], batch_data_14.Vang_out_abc[:,3,:],
                    batch_data_15.Vang_out_abc[:,3,:], batch_data_16.Vang_out_abc[:,3,:],
                    batch_data_17.Vang_out_abc[:,3,:], batch_data_18.Vang_out_abc[:,3,:])

    Pd_out_a = hcat(batch_data_1.Pd_out_abc[:,1,:], batch_data_2.Pd_out_abc[:,1,:],
                    batch_data_3.Pd_out_abc[:,1,:], batch_data_4.Pd_out_abc[:,1,:],
                    batch_data_5.Pd_out_abc[:,1,:], batch_data_6.Pd_out_abc[:,1,:],
                    batch_data_7.Pd_out_abc[:,1,:]  , batch_data_8.Pd_out_abc[:,1,:],
                    batch_data_9.Pd_out_abc[:,1,:], batch_data_10.Pd_out_abc[:,1,:],
                    batch_data_11.Pd_out_abc[:,1,:], batch_data_12.Pd_out_abc[:,1,:],
                    batch_data_13.Pd_out_abc[:,1,:], batch_data_14.Pd_out_abc[:,1,:],
                    batch_data_15.Pd_out_abc[:,1,:], batch_data_16.Pd_out_abc[:,1,:],
                    batch_data_17.Pd_out_abc[:,1,:], batch_data_18.Pd_out_abc[:,1,:])

    Pd_out_b = hcat(batch_data_1.Pd_out_abc[:,2,:], batch_data_2.Pd_out_abc[:,2,:],
                    batch_data_3.Pd_out_abc[:,2,:], batch_data_4.Pd_out_abc[:,2,:],
                    batch_data_5.Pd_out_abc[:,2,:], batch_data_6.Pd_out_abc[:,2,:],
                    batch_data_7.Pd_out_abc[:,2,:], batch_data_8.Pd_out_abc[:,2,:],
                    batch_data_9.Pd_out_abc[:,2,:], batch_data_10.Pd_out_abc[:,2,:],
                    batch_data_11.Pd_out_abc[:,2,:], batch_data_12.Pd_out_abc[:,2,:],
                    batch_data_13.Pd_out_abc[:,2,:], batch_data_14.Pd_out_abc[:,2,:],
                    batch_data_15.Pd_out_abc[:,2,:], batch_data_16.Pd_out_abc[:,2,:],
                    batch_data_17.Pd_out_abc[:,2,:], batch_data_18.Pd_out_abc[:,2,:])

    Pd_out_c = hcat(batch_data_1.Pd_out_abc[:,3,:], batch_data_2.Pd_out_abc[:,3,:],
                    batch_data_3.Pd_out_abc[:,3,:], batch_data_4.Pd_out_abc[:,3,:],
                    batch_data_5.Pd_out_abc[:,3,:], batch_data_6.Pd_out_abc[:,3,:],
                    batch_data_7.Pd_out_abc[:,3,:], batch_data_8.Pd_out_abc[:,3,:],
                    batch_data_9.Pd_out_abc[:,3,:], batch_data_10.Pd_out_abc[:,3,:],
                    batch_data_11.Pd_out_abc[:,3,:], batch_data_12.Pd_out_abc[:,3,:],
                    batch_data_13.Pd_out_abc[:,3,:], batch_data_14.Pd_out_abc[:,3,:],
                    batch_data_15.Pd_out_abc[:,3,:], batch_data_16.Pd_out_abc[:,3,:],
                    batch_data_17.Pd_out_abc[:,3,:], batch_data_18.Pd_out_abc[:,3,:])

    Qd_out_a = hcat(batch_data_1.Qd_out_abc[:,1,:], batch_data_2.Qd_out_abc[:,1,:],
                    batch_data_3.Qd_out_abc[:,1,:], batch_data_4.Qd_out_abc[:,1,:],
                    batch_data_5.Qd_out_abc[:,1,:], batch_data_6.Qd_out_abc[:,1,:],
                    batch_data_7.Qd_out_abc[:,1,:], batch_data_8.Qd_out_abc[:,1,:],
                    batch_data_9.Qd_out_abc[:,1,:], batch_data_10.Qd_out_abc[:,1,:],
                    batch_data_11.Qd_out_abc[:,1,:], batch_data_12.Qd_out_abc[:,1,:],
                    batch_data_13.Qd_out_abc[:,1,:], batch_data_14.Qd_out_abc[:,1,:],
                    batch_data_15.Qd_out_abc[:,1,:], batch_data_16.Qd_out_abc[:,1,:],
                    batch_data_17.Qd_out_abc[:,1,:], batch_data_18.Qd_out_abc[:,1,:])

    Qd_out_b = hcat(batch_data_1.Qd_out_abc[:,2,:], batch_data_2.Qd_out_abc[:,2,:],
                    batch_data_3.Qd_out_abc[:,2,:], batch_data_4.Qd_out_abc[:,2,:],
                    batch_data_5.Qd_out_abc[:,2,:], batch_data_6.Qd_out_abc[:,2,:],
                    batch_data_7.Qd_out_abc[:,2,:], batch_data_8.Qd_out_abc[:,2,:],
                    batch_data_9.Qd_out_abc[:,2,:], batch_data_10.Qd_out_abc[:,2,:],
                    batch_data_11.Qd_out_abc[:,2,:], batch_data_12.Qd_out_abc[:,2,:],
                    batch_data_13.Qd_out_abc[:,2,:], batch_data_14.Qd_out_abc[:,2,:],
                    batch_data_15.Qd_out_abc[:,2,:], batch_data_16.Qd_out_abc[:,2,:],
                    batch_data_17.Qd_out_abc[:,2,:], batch_data_18.Qd_out_abc[:,2,:])

    Qd_out_c = hcat(batch_data_1.Qd_out_abc[:,3,:], batch_data_2.Qd_out_abc[:,3,:],
                    batch_data_3.Qd_out_abc[:,3,:], batch_data_4.Qd_out_abc[:,3,:],
                    batch_data_5.Qd_out_abc[:,3,:], batch_data_6.Qd_out_abc[:,3,:],
                    batch_data_7.Qd_out_abc[:,3,:], batch_data_8.Qd_out_abc[:,3,:],
                    batch_data_9.Qd_out_abc[:,3,:], batch_data_10.Qd_out_abc[:,3,:],
                    batch_data_11.Qd_out_abc[:,3,:], batch_data_12.Qd_out_abc[:,3,:],
                    batch_data_13.Qd_out_abc[:,3,:], batch_data_14.Qd_out_abc[:,3,:],
                    batch_data_15.Qd_out_abc[:,3,:], batch_data_16.Qd_out_abc[:,3,:],
                    batch_data_17.Qd_out_abc[:,3,:], batch_data_18.Qd_out_abc[:,3,:])

    # Pg_out_abc = hcat(batch_data_1.Pg_out_abc, batch_data_2.Pg_out_abc,
    #                 batch_data_3.Pg_out_abc, batch_data_4.Pg_out_abc,
    #                 batch_data_5.Pg_out_abc, batch_data_6.Pg_out_abc,
    #                 batch_data_7.Pg_out_abc, batch_data_8.Pg_out_abc,
    #                 batch_data_9.Pg_out_abc, batch_data_10.Pg_out_abc,
    #                 batch_data_11.Pg_out_abc, batch_data_12.Pg_out_abc,
    #                 batch_data_13.Pg_out_abc, batch_data_14.Pg_out_abc,
    #                 batch_data_15.Pg_out_abc, batch_data_16.Pg_out_abc,
    #                 batch_data_17.Pg_out_abc, batch_data_18.Pg_out_abc)

    # Qg_out_abc = hcat(batch_data_1.Qg_out_abc, batch_data_2.Qg_out_abc,
    #                 batch_data_3.Qg_out_abc, batch_data_4.Qg_out_abc,
    #                 batch_data_5.Qg_out_abc, batch_data_6.Qg_out_abc,
    #                 batch_data_7.Qg_out_abc, batch_data_8.Qg_out_abc,
    #                 batch_data_9.Qg_out_abc, batch_data_10.Qg_out_abc,
    #                 batch_data_11.Qg_out_abc, batch_data_12.Qg_out_abc,
    #                 batch_data_13.Qg_out_abc, batch_data_14.Qg_out_abc,
    #                 batch_data_15.Qg_out_abc, batch_data_16.Qg_out_abc,
    #                 batch_data_17.Qg_out_abc, batch_data_18.Qg_out_abc)

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