# Connect check to Gazelle board (direct LAN, no tunnel). ASCII only for PS5.1.
$BOARD = '192.168.31.158'
Write-Host '== 1. local LAN IP =='
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '192.168.31.*' } |
  ForEach-Object { Write-Host ('  ' + $_.IPAddress + '  ' + $_.InterfaceAlias) }

Write-Host '== 2. TCP ports =='
foreach ($p in @(22,8000)) {
  $r = Test-NetConnection -ComputerName $BOARD -Port $p -WarningAction SilentlyContinue
  $st = $(if ($r.TcpTestSucceeded) {'OPEN'} else {'CLOSE / not reachable'})
  Write-Host ('  :' + $p + '  ' + $st)
}

Write-Host '== 3. optical server /health =='
try {
  $h = Invoke-RestMethod -Uri ('http://' + $BOARD + ':8000/health') -TimeoutSec 5
  Write-Host ('  status=' + $h.status + '  tia_gain=' + $h.tia_gain + '  calls=' + $h.calls + '  cached=' + $h.cached_weights + '  uptime=' + $h.uptime_s + 's')
} catch {
  Write-Host ('  /health FAILED: ' + $_.Exception.Message + '  (server_gazelle not running?)')
}

Write-Host '== 4. matmul probe (1x2 @ 2x1, expect [[-1]]) =='
try {
  $body = '{"act_b64":"AQI=","act_shape":[1,2],"weight_b64":"Af8=","weight_shape":[2,1]}'
  $m = Invoke-RestMethod -Uri ('http://' + $BOARD + ':8000/matmul') -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host ('  matmul returned: ' + ($m.data -join ','))
} catch {
  Write-Host ('  matmul FAILED: ' + $_.Exception.Message)
}

Write-Host '== done. If 8000 OPEN and matmul=-1, board ready. =='