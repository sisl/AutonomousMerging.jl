
using Documenter, AutonomousMerging

makedocs(
	 modules = [AutonomousMerging],
	 format = Documenter.HTML(),
	 sitename="AutonomousMerging.jl"
	 )

deploydocs(
    repo = "github.com/sisl/AutonomousMerging.jl.git",
)