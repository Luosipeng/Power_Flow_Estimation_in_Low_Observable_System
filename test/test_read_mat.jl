include("../ios/read_mat.jl")
using MAT

batch_path_1 = "D:/luosipeng/matpower8.1/pf_parallel_out/batch_001.mat"
batch_path_2 = "D:/luosipeng/matpower8.1/pf_parallel_out/batch_002.mat"
batch_path_4 = "D:/luosipeng/matpower8.1/pf_parallel_out/batch_004.mat"


batch_data_1 = read_batch_mat(batch_path_1)
batch_data_2 = read_batch_mat(batch_path_2)
batch_data_4 = read_batch_mat(batch_path_4)

# branch = read_topology_mat("D:/luosipeng/DistributionPowerFlow/topology.mat")
Vmag_abc = batch_data_1.Vmag_out_abc[:,:,1]
Vang_abc = batch_data_1.Vang_out_abc[:,:,1]

