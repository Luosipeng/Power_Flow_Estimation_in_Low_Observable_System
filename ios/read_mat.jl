using MAT
function read_mat()
    # Input file paths
    # batch_path_1 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_001.mat"
    # batch_path_2 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_002.mat"
    # batch_path_3 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_003.mat"
    # batch_path_4 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_004.mat"
    # batch_path_5 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_005.mat"
    # batch_path_6 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_006.mat"
    # batch_path_7 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_007.mat"
    # batch_path_8 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_008.mat"
    # batch_path_9 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_009.mat"
    # batch_path_10 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_010.mat"
    # batch_path_11 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_011.mat"
    # batch_path_12 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_012.mat"
    # batch_path_13 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_013.mat"
    # batch_path_14 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_014.mat"
    # batch_path_15 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_015.mat"
    # batch_path_16 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_016.mat"
    # batch_path_17 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_017.mat"
    # batch_path_18 = "/home/user/Downloads/Distribution_System_State_Estimation-main/pf_out/batch_018.mat"


    # batch_path_1 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_001.mat"
    # batch_path_2 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_002.mat"
    # batch_path_3 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_003.mat"
    # batch_path_4 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_004.mat"
    # batch_path_5 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_005.mat"
    # batch_path_6 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_006.mat"
    # batch_path_7 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_007.mat"
    # batch_path_8 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_008.mat"
    # batch_path_9 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_009.mat"
    # batch_path_10 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_010.mat"
    # batch_path_11 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_011.mat"
    # batch_path_12 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_012.mat"
    # batch_path_13 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_013.mat"
    # batch_path_14 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_014.mat"
    # batch_path_15 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_015.mat"
    # batch_path_16 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_016.mat"
    # batch_path_17 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_017.mat"
    # batch_path_18 = "/home/user/Downloads/powerflow_parallel/batch_ieee123_018.mat"

    batch_path_1 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_001.mat"
    batch_path_2 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_002.mat"
    batch_path_3 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_003.mat"
    batch_path_4 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_004.mat"
    batch_path_5 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_005.mat"
    batch_path_6 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_006.mat"
    batch_path_7 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_007.mat"
    batch_path_8 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_008.mat"
    batch_path_9 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_009.mat"
    batch_path_10 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_010.mat"
    batch_path_11 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_011.mat"
    batch_path_12 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_012.mat"
    batch_path_13 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_013.mat"
    batch_path_14 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_014.mat"
    batch_path_15 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_015.mat"
    batch_path_16 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_016.mat"
    batch_path_17 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_017.mat"
    batch_path_18 = "D:/luosipeng/Code_Script/Linux_shared_project/power_flow_data/batch_ieee123_018.mat"

    # Read MAT files
    batch_data_1 = read_batch_mat(batch_path_1)
    batch_data_2 = read_batch_mat(batch_path_2)
    batch_data_3 = read_batch_mat(batch_path_3)
    batch_data_4 = read_batch_mat(batch_path_4)
    batch_data_5 = read_batch_mat(batch_path_5)
    batch_data_6 = read_batch_mat(batch_path_6)
    batch_data_7 = read_batch_mat(batch_path_7)
    batch_data_8 = read_batch_mat(batch_path_8)
    batch_data_9 = read_batch_mat(batch_path_9)
    batch_data_10 = read_batch_mat(batch_path_10)
    batch_data_11 = read_batch_mat(batch_path_11)
    batch_data_12 = read_batch_mat(batch_path_12)
    batch_data_13 = read_batch_mat(batch_path_13)
    batch_data_14 = read_batch_mat(batch_path_14)
    batch_data_15 = read_batch_mat(batch_path_15)
    batch_data_16 = read_batch_mat(batch_path_16)
    batch_data_17 = read_batch_mat(batch_path_17)
    batch_data_18 = read_batch_mat(batch_path_18)

    return (batch_data_1, batch_data_2, batch_data_3, batch_data_4,
            batch_data_5, batch_data_6, batch_data_7, batch_data_8,
            batch_data_9, batch_data_10, batch_data_11, batch_data_12,
            batch_data_13, batch_data_14, batch_data_15, batch_data_16, batch_data_17,
            batch_data_18)
end

function read_batch_mat(path::AbstractString)
    mat = matopen(path)
    try
        # Vmag_out = read(mat, "Vmag_out")
        # Vang_out = read(mat, "Vang_out")
        # Pg_out = read(mat, "Pg_out")
        # Qg_out = read(mat, "Qg_out")
        # Pd_out = read(mat, "Pd_out")
        # Qd_out = read(mat, "Qd_out")
        Vmag_out_abc = read(mat, "Vmag_out_abc")
        Vang_out_abc = read(mat, "Vang_out_abc")
        Pg_out_abc = read(mat, "Pg_out_abc")
        Qg_out_abc = read(mat, "Qg_out_abc")
        Pd_out_abc = read(mat, "Pd_out_abc")
        Qd_out_abc = read(mat, "Qd_out_abc")

        success  = read(mat, "success")


        return (; Vmag_out_abc, Vang_out_abc, Pg_out_abc, Qg_out_abc, Pd_out_abc, Qd_out_abc, success)
    finally
        close(mat)
    end
end

function read_topology_mat(path::AbstractString)
    mat = matopen(path)
    try
        branch_ext_raw = read(mat, "branch_ext_raw")
        return branch_ext_raw
    finally
        close(mat)
    end
end