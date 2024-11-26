using MX2RKKY
using Documenter

DocMeta.setdocmeta!(MX2RKKY, :DocTestSetup, :(using MX2RKKY); recursive=true)

makedocs(;
    modules=[MX2RKKY],
    authors="Igor Krivenko <iskrivenko@proton.me> and contributors",
    sitename="MX2RKKY.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
