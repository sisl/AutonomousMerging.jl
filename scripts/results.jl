using PGFPlots
using CSV
using DataFrames


res = CSV.read("results.csv")

pcoll = Axis()
for (i,pol) in enumerate(res[:policy])
    push!(pcoll, Plots.BarChart([Float64.(res[:c_rate][i])], style="blue", legendentry=pol))
end



pavg = Axis(Plots.BarChart(res[:policy], Float64.(res[:avg_steps])),
            ylabel = "Average steps")


ptout = Axis(Plots.BarChart(res[:policy], Float64.(res[:t_out])),
             ylabel = "Percentage of time out failures")

g= GroupPlot(1,3, groupStyle="horizontal sep=1cm, vertical sep=1cm");
push!(g, pcoll)
push!(g, pavg)
push!(g, ptout)

save("barplots.tex", g)