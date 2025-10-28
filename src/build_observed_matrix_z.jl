function build_observed_matrix_Z(daily_predictions; monitor_buses::AbstractSet{Int}=Set{Int}())
    sensors = get(daily_predictions, "sensors", Dict{String, Any}())
    num_nodes, num_cols = 33, 5
    column_map = Dict(
        "p" => 1, "pb" => 1,
        "q" => 2, "qb" => 2,
        "vreal" => 3,
        "vimag" => 4,
        "vmag" => 5, "vabs" => 5, "vmagnitude" => 5
    )

    Z = zeros(Float32, num_nodes, num_cols)
    observed_pairs = Tuple{Int,Int}[]
    monitored_obs = Dict{Int, Dict{Int, Float32}}()  # node_idx => (col => value)

    for (name, data) in sensors
        parts = split(name, '-')
        length(parts) < 3 && continue

        node_idx = try
            parse(Int, parts[2])
        catch
            continue
        end
        (1 <= node_idx <= num_nodes) || continue

        token = lowercase(replace(parts[end], r"[\s_()]" => ""))
        haskey(column_map, token) || continue
        col = column_map[token]

        preds = get(data, "prediction_mean", Float32[])
        isempty(preds) && continue
        val = Float32(preds[1])

        if node_idx in monitor_buses
            # 仅记录到 monitored_obs，不写入 Z/observed_pairs
            inner = get!(monitored_obs, node_idx, Dict{Int, Float32}())
            inner[col] = val
            continue
        end

        Z[node_idx, col] = val
        push!(observed_pairs, (node_idx, col))
    end

    return Z, observed_pairs, monitored_obs
end
