param(
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "docs\refs"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$references = @(
    @{ Name = "vaswani2017_attention_is_all_you_need.pdf"; Url = "https://arxiv.org/pdf/1706.03762" },
    @{ Name = "yang2016_hierarchical_attention_networks.pdf"; Url = "https://aclanthology.org/N16-1174.pdf" },
    @{ Name = "beltagy2020_longformer.pdf"; Url = "https://arxiv.org/pdf/2004.05150" },
    @{ Name = "zaheer2020_bigbird.pdf"; Url = "https://arxiv.org/pdf/2007.14062" },
    @{ Name = "kitaev2020_reformer.pdf"; Url = "https://arxiv.org/pdf/2001.04451" },
    @{ Name = "roy2021_routing_transformer.pdf"; Url = "https://arxiv.org/pdf/2003.05997" },
    @{ Name = "rae2020_compressive_transformer.pdf"; Url = "https://arxiv.org/pdf/1911.05507" },
    @{ Name = "wu2022_memorizing_transformers.pdf"; Url = "https://arxiv.org/pdf/2203.08913" },
    @{ Name = "borgeaud2022_retro.pdf"; Url = "https://arxiv.org/pdf/2112.04426" },
    @{ Name = "dehghani2019_universal_transformers.pdf"; Url = "https://arxiv.org/pdf/1807.03819" },
    @{ Name = "graves2016_adaptive_computation_time.pdf"; Url = "https://arxiv.org/pdf/1603.08983" },
    @{ Name = "lewis2020_retrieval_augmented_generation.pdf"; Url = "https://arxiv.org/pdf/2005.11401" },
    @{ Name = "guu2020_realm.pdf"; Url = "https://arxiv.org/pdf/2002.08909" },
    @{ Name = "karpukhin2020_dense_passage_retrieval.pdf"; Url = "https://arxiv.org/pdf/2004.04906" },
    @{ Name = "yao2023_react.pdf"; Url = "https://arxiv.org/pdf/2210.03629" },
    @{ Name = "schick2023_toolformer.pdf"; Url = "https://arxiv.org/pdf/2302.04761" },
    @{ Name = "kwon2023_vllm_pagedattention.pdf"; Url = "https://arxiv.org/pdf/2309.06180" }
)

foreach ($ref in $references) {
    $target = Join-Path $OutputDir $ref.Name
    if (Test-Path $target) {
        Write-Host "exists $($ref.Name)"
        continue
    }
    Write-Host "downloading $($ref.Name)"
    Invoke-WebRequest -Uri $ref.Url -OutFile $target -MaximumRedirection 5
}

Write-Host "Downloaded references to $OutputDir"
