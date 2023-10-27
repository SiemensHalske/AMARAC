Add-Type -AssemblyName System.Windows.Forms

# GUI Elements
$form = New-Object System.Windows.Forms.Form
$form.Text = 'Network Scanner'
$form.Size = New-Object System.Drawing.Size(300,200)

$button = New-Object System.Windows.Forms.Button
$button.Text = 'Start Scan'
$button.Width = 100
$button.Height = 30
$button.Top = 50
$button.Left = 100

# Action when button is clicked
$button.Add_Click({
    $results = @()
    $startIP = [IPAddress]::Parse("192.168.1.1").Address
    $endIP = [IPAddress]::Parse("192.168.1.254").Address

    for ($ip = $startIP; $ip -le $endIP; $ip++) {
        $ipStr = ([IPAddress]$ip).IPAddressToString
        $ports = @(9100, 443)
        foreach ($port in $ports) {
            $tcpClient = New-Object System.Net.Sockets.TcpClient
            $connect = $tcpClient.BeginConnect($ipStr, $port, $null, $null)
            $success = $connect.AsyncWaitHandle.WaitOne(500, $true)  # 500ms timeout
            if ($success) {
                $tcpClient.EndConnect($connect)
                $results += [PSCustomObject]@{
                    IPAddress = $ipStr
                    Port      = $port
                }
            }
            $tcpClient.Close()
        }
    }
    $results | Export-Csv -Path "NetworkDevices.csv" -NoTypeInformation
})

$form.Controls.Add($button)

# Show the form
$form.ShowDialog()
