# 演示机侧: 连内网后先跑这个确认到板连通 + 8000 光算服务健康
$BOARD = '192.168.31.158'
Write-Host '== 1. 本机内网 IP =='
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '192.168.31.*' } |
  ForEach-Object { Write-Host ("  " + $_.IPAddress + "  " + $_.InterfaceAlias) }

Write-Host '`n== 2. TCP 探测 =='
foreach ($p in @(22,8000)) {
  $r = Test-NetConnection -ComputerName $BOARD -Port $p -WarningAction SilentlyContinue
  Write-Host ("  :{0}  {1}" -f $p, $(if ($r.TcpTestSucceeded) {'OPEN'} else {'CLOSE/不通'}))
}

Write-Host '`n== 3. 光算服务 /health (可直达, 无需隧道) =='
try {
  $h = Invoke-RestMethod -Uri "http://$BOARD`:8000/health" -TimeoutSec 5
  Write-Host ("  status={0}  tia_gain={1}  calls={2}  cached={3}  uptime={4}s" -f $h.status, $h.tia_gain, $h.calls, $h.cached_weights, $h.uptime_s)
} catch {
  Write-Host "  /health 失败: $($_.Exception.Message)  (板上 server_gazelle 未起?)"
}

Write-Host '`n== 4. matmul 探针 (1x2 @ 2x1, 期望 [[-1]]) =='
try {
  $body = '{"act_b64":"AQI=","act_shape":[1,2],"weight_b64":"Af8=","weight_shape":[2,1]}'
  $m = Invoke-RestMethod -Uri "http://$BOARD`:8000/matmul" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
  Write-Host ("  matmul 返回: {0}" -f ($m.data -join ','))
} catch {
  Write-Host "  matmul 失败: $($_.Exception.Message)"
}

if ($LASTEXITCODE -eq 0) {
  Write-Host '`n结论: 内网连通正常, 可启动 demo 前端。'
}