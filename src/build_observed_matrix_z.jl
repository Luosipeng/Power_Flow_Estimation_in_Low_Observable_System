function build_observed_matrix_Z(daily_predictions)
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
        Z[node_idx, col] = Float32(preds[1])
        push!(observed_pairs, (node_idx, col))
    end

    return Z, observed_pairs
end