Param(
    [string[]]$Activations = @('sigmoid','tanh','relu','leaky_relu'),
    [int[]]$Hiddens = @(16,32,64),
    [int[]]$Batches = @(64,128,256),
    [int]$Epochs = 50,
    [double]$Lr = 0.001,
    [string]$Python = 'python',
    [string]$Mpiexec = 'mpiexec',
    [int]$Np = 4
)

# Example usage:
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Activations sigmoid,tanh -Hiddens 32,64 -Batches 128 -Epochs 100 -Lr 0.001 -Np 4

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptPath

foreach ($act in $Activations) {
    foreach ($h in $Hiddens) {
        foreach ($b in $Batches) {
            Write-Host "Running act=$act hidden=$h batch=$b epochs=$Epochs lr=$Lr np=$Np"
            $cmd = "$Mpiexec -n $Np $Python sgd.py --activation $act --hidden $h --batch $b --epochs $Epochs --lr $Lr"
            Write-Host $cmd
            try {
                & $Mpiexec -n $Np $Python sgd.py --activation $act --hidden $h --batch $b --epochs $Epochs --lr $Lr
            }
            catch {
                Write-Warning "Run failed: act=$act hidden=$h batch=$b. Error: $_"
            }
        }
    }
}

Pop-Location
