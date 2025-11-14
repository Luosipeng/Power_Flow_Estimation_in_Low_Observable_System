include("../ios/read_mat.jl")
using MAT

# batch_path_1 = "D:/luosipeng/matpower8.1/pf_parallel_out/batch_001.mat"
# batch_path_2 = "D:/luosipeng/matpower8.1/pf_parallel_out/batch_002.mat"
# batch_path_3 = "D:/luosipeng/matpower8.1/pf_parallel_out/batch_003.mat"


# batch_data_1 = read_batch_mat(batch_path_1)
# batch_data_2 = read_batch_mat(batch_path_2)
# batch_data_3 = read_batch_mat(batch_path_3)

# branch = read_topology_mat("D:/luosipeng/matpower8.1/pf_parallel_out/topology.mat")

# V = batch_data_1.Vmag_out[:,1]

batch_path_1 = "C:/Users/PC/Desktop/paper_case/results_thread_1.mat"
batch_path_2 = "C:/Users/PC/Desktop/paper_case/results_thread_2.mat"
batch_path_4 = "C:/Users/PC/Desktop/paper_case/results_thread_4.mat"


batch_data_1 = read_batch_mat(batch_path_1)
batch_data_2 = read_batch_mat(batch_path_2)
batch_data_4 = read_batch_mat(batch_path_4)

# branch = read_topology_mat("D:/luosipeng/DistributionPowerFlow/topology.mat")
Vmag = batch_data_1.Vmag_out_ac[:,1]
Vmag_dc = batch_data_1.vmag_out_dc[:,1]

Pd = batch_data_1.Pd_out_dc[:,1]
Pd_ac = batch_data_1.Pd_out_ac[:,1]

Vmag_dc = batch_data_4.vmag_out_dc[:,30000]