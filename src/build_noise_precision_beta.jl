function build_noise_precision_beta(daily_predictions)
    sensors = get(daily_predictions, "sensors", Dict{String, Any}())
    num_nodes, num_cols = 37, 5
    column_map = Dict(
        "p" => 1, "pb" => 1,
        "q" => 2, "qb" => 2,
        "vreal" => 3,
        "vimag" => 4,
        "vmag" => 5, "vabs" => 5, "vmagnitude" => 5
    )

    β = zeros(Float32, num_nodes, num_cols)

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
        stds = get(data, "prediction_std", Float32[])
        isempty(stds) && continue
        σ = abs(Float32(stds[1]))
        σ == 0f0 && continue
        β[node_idx, col] = 1f0 / (σ^2)
    end

    return β
end