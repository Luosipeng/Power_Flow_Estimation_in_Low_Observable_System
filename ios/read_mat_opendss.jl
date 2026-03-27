using MAT
function read_mat_opendss(path::AbstractString)
    mat = matopen(path)
    try
        # Vmag_out = read(mat, "Vmag_out")
        # Vang_out = read(mat, "Vang_out")
        # Pg_out = read(mat, "Pg_out")
        # Qg_out = read(mat, "Qg_out")
        # Pd_out = read(mat, "Pd_out")
        # Qd_out = read(mat, "Qd_out")
        data = read(mat, "data")


        return data
    finally
        close(mat)
    end
end

function read_ybus_opendss()
    mat = matopen("D:/Linux_shared_folder/topology.mat")
    try
        Y_bus = read(mat, "Ybus")
        return Y_bus
    finally
        close(mat)
    end
end

function read_all_opendss_data()
    # Voltage Information
    path_voltage_mag_a = "D:/Linux_shared_folder/voltage_mag_a.mat"
    path_voltage_mag_b = "D:/Linux_shared_folder/voltage_mag_b.mat"
    path_voltage_mag_c = "D:/Linux_shared_folder/voltage_mag_c.mat"

    path_voltage_ang_a = "D:/Linux_shared_folder/voltage_ang_a.mat"
    path_voltage_ang_b = "D:/Linux_shared_folder/voltage_ang_b.mat"
    path_voltage_ang_c = "D:/Linux_shared_folder/voltage_ang_c.mat"
    voltage_mag_a = read_mat_opendss(path_voltage_mag_a)
    voltage_mag_b = read_mat_opendss(path_voltage_mag_b)
    voltage_mag_c = read_mat_opendss(path_voltage_mag_c)

    voltage_ang_a = read_mat_opendss(path_voltage_ang_a)
    voltage_ang_b = read_mat_opendss(path_voltage_ang_b)
    voltage_ang_c = read_mat_opendss(path_voltage_ang_c)

    # Power Information
    path_power_p_a = "D:/Linux_shared_folder/injection_p_a.mat"
    path_power_p_b = "D:/Linux_shared_folder/injection_p_b.mat"
    path_power_p_c = "D:/Linux_shared_folder/injection_p_c.mat"

    path_power_q_a = "D:/Linux_shared_folder/injection_q_a.mat"
    path_power_q_b = "D:/Linux_shared_folder/injection_q_b.mat"
    path_power_q_c = "D:/Linux_shared_folder/injection_q_c.mat"
    power_p_a = read_mat_opendss(path_power_p_a)
    power_p_b = read_mat_opendss(path_power_p_b)
    power_p_c = read_mat_opendss(path_power_p_c)
    power_q_a = read_mat_opendss(path_power_q_a)
    power_q_b = read_mat_opendss(path_power_q_b)
    power_q_c = read_mat_opendss(path_power_q_c)


    # Topology Information
    # path_topology = "D:/luosipeng/DistributionPowerFlow/data/opendss/123Bus/topology.mat"
    # topology = read_mat_opendss(path_topology)

    return (voltage_mag_a, voltage_mag_b, voltage_mag_c,
            voltage_ang_a, voltage_ang_b, voltage_ang_c,
            power_p_a, power_p_b, power_p_c,
            power_q_a, power_q_b, power_q_c)
end