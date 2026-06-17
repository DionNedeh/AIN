# Terminal Activity Feed
# A terminal-style live Windows activity monitor.
# Run in Windows PowerShell as Administrator for best results.

[CmdletBinding()]
param(
    [string[]]$WatchPaths = @(),

    [ValidateRange(1, 60)]
    [int]$NetworkPollSeconds = 2,

    [switch]$NoProcesses,
    [switch]$NoFiles,
    [switch]$NoNetwork,
    [switch]$NoCleanup
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RunId = "TAF_{0}_{1}" -f $PID, ([guid]::NewGuid().ToString("N").Substring(0, 8))
$script:Subscriptions = @()
$script:Watchers = @()
$script:SeenConnections = @{}
$script:LastNetworkPoll = Get-Date

function Write-Feed {
    param(
        [string]$Type,
        [string]$Message,
        [ConsoleColor]$Color = "Gray"
    )

    $time = (Get-Date).ToString("HH:mm:ss.fff")
    Write-Host "[$time] [$Type] $Message" -ForegroundColor $Color
}

function Get-DefaultWatchPaths {
    $candidates = @()

    $desktop = [Environment]::GetFolderPath("Desktop")
    if (-not [string]::IsNullOrWhiteSpace($desktop)) {
        $candidates += $desktop
    }

    if ($env:USERPROFILE) {
        $candidates += Join-Path $env:USERPROFILE "OneDrive\Desktop"
        $candidates += Join-Path $env:USERPROFILE "Desktop"
        $candidates += Join-Path $env:USERPROFILE "Downloads"
    }

    if ($env:TEMP) {
        $candidates += $env:TEMP
    }

    $resolved = New-Object System.Collections.Generic.List[string]

    foreach ($path in $candidates) {
        if (Test-Path $path) {
            try {
                $fullPath = (Resolve-Path $path).Path
                if (-not $resolved.Contains($fullPath)) {
                    [void]$resolved.Add($fullPath)
                }
            } catch {
                # Ignore paths that cannot be resolved.
            }
        }
    }

    return $resolved.ToArray()
}

function Test-ActivityFeedSourceId {
    param([string]$SourceIdentifier)

    if ([string]::IsNullOrWhiteSpace($SourceIdentifier)) {
        return $false
    }

    return (
        $SourceIdentifier -like "TAF_*" -or
        $SourceIdentifier -eq "PROC_START" -or
        $SourceIdentifier -eq "PROC_STOP" -or
        $SourceIdentifier -like "FILE_CREATED_*" -or
        $SourceIdentifier -like "FILE_CHANGED_*" -or
        $SourceIdentifier -like "FILE_DELETED_*" -or
        $SourceIdentifier -like "FILE_RENAMED_*"
    )
}

function Clear-StaleActivityFeedEvents {
    $oldSubscribers = Get-EventSubscriber -ErrorAction SilentlyContinue | Where-Object {
        Test-ActivityFeedSourceId -SourceIdentifier $_.SourceIdentifier
    }

    foreach ($subscriber in $oldSubscribers) {
        Unregister-Event -SubscriptionId $subscriber.SubscriptionId -ErrorAction SilentlyContinue
    }

    Get-Event -ErrorAction SilentlyContinue | Where-Object {
        Test-ActivityFeedSourceId -SourceIdentifier $_.SourceIdentifier
    } | Remove-Event -ErrorAction SilentlyContinue
}

function Get-ProcessNameSafe {
    param([int]$TargetProcessId)

    try {
        return (Get-Process -Id $TargetProcessId -ErrorAction Stop).ProcessName
    } catch {
        return "unknown"
    }
}

function Get-TcpSnapshot {
    $snapshot = @{}

    try {
        $connections = Get-NetTCPConnection -ErrorAction Stop |
            Where-Object {
                $_.State -in @("Established", "Listen", "SynSent", "SynReceived")
            }

        foreach ($connection in $connections) {
            $key = "$($connection.OwningProcess)|$($connection.LocalAddress):$($connection.LocalPort)|$($connection.RemoteAddress):$($connection.RemotePort)|$($connection.State)"
            $snapshot[$key] = $connection
        }
    } catch {
        Write-Feed "WARN" "Could not read TCP connections. Try running as Administrator." Yellow
    }

    return $snapshot
}

function Poll-Network {
    if ($NoNetwork) {
        return
    }

    $current = Get-TcpSnapshot

    foreach ($key in $current.Keys) {
        if (-not $script:SeenConnections.ContainsKey($key)) {
            $connection = $current[$key]
            $processName = Get-ProcessNameSafe -TargetProcessId $connection.OwningProcess

            Write-Feed "NET+" "$processName PID=$($connection.OwningProcess) $($connection.LocalAddress):$($connection.LocalPort) -> $($connection.RemoteAddress):$($connection.RemotePort) $($connection.State)" Magenta
        }
    }

    foreach ($key in @($script:SeenConnections.Keys)) {
        if (-not $current.ContainsKey($key)) {
            $connection = $script:SeenConnections[$key]
            $processName = Get-ProcessNameSafe -TargetProcessId $connection.OwningProcess

            Write-Feed "NET-" "$processName PID=$($connection.OwningProcess) $($connection.LocalAddress):$($connection.LocalPort) -> $($connection.RemoteAddress):$($connection.RemotePort) closed" DarkMagenta
        }
    }

    $script:SeenConnections = $current
}

function Handle-ActivityEvent {
    param($Event)

    switch -Regex ($Event.SourceIdentifier) {
        "_PROC_START$" {
            $eventData = $Event.SourceEventArgs.NewEvent
            Write-Feed "PROC+" "$($eventData.ProcessName) PID=$($eventData.ProcessID) ParentPID=$($eventData.ParentProcessID)" Green
            break
        }

        "_PROC_STOP$" {
            $eventData = $Event.SourceEventArgs.NewEvent
            Write-Feed "PROC-" "$($eventData.ProcessName) PID=$($eventData.ProcessID)" DarkGreen
            break
        }

        "_FILE_CREATED_\d+$" {
            $eventData = $Event.SourceEventArgs
            Write-Feed "FILE+" "$($eventData.FullPath)" Cyan
            break
        }

        "_FILE_CHANGED_\d+$" {
            $eventData = $Event.SourceEventArgs
            Write-Feed "FILE*" "$($eventData.FullPath)" DarkCyan
            break
        }

        "_FILE_DELETED_\d+$" {
            $eventData = $Event.SourceEventArgs
            Write-Feed "FILE-" "$($eventData.FullPath)" Red
            break
        }

        "_FILE_RENAMED_\d+$" {
            $eventData = $Event.SourceEventArgs
            Write-Feed "FILE>" "$($eventData.OldFullPath) -> $($eventData.FullPath)" Yellow
            break
        }
    }
}

try {
    Clear-Host
    Write-Feed "BOOT" "Starting Terminal Activity Feed..." White

    if (-not $NoCleanup) {
        Clear-StaleActivityFeedEvents
    }

    if (-not $NoProcesses) {
        $script:Subscriptions += Register-WmiEvent -Class Win32_ProcessStartTrace -SourceIdentifier "${script:RunId}_PROC_START"
        $script:Subscriptions += Register-WmiEvent -Class Win32_ProcessStopTrace  -SourceIdentifier "${script:RunId}_PROC_STOP"
        Write-Feed "WATCH" "Watching process start/stop events" Blue
    }

    if (-not $NoFiles) {
        if ($WatchPaths.Count -eq 0) {
            $WatchPaths = Get-DefaultWatchPaths
        }

        $index = 0

        foreach ($path in $WatchPaths) {
            if (Test-Path $path) {
                $resolved = (Resolve-Path $path).Path

                $watcher = New-Object System.IO.FileSystemWatcher
                $watcher.Path = $resolved
                $watcher.IncludeSubdirectories = $true
                $watcher.EnableRaisingEvents = $true
                $watcher.NotifyFilter = [System.IO.NotifyFilters]"FileName, DirectoryName, LastWrite, Size, CreationTime"

                $script:Watchers += $watcher

                $script:Subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Created -SourceIdentifier "${script:RunId}_FILE_CREATED_$index"
                $script:Subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Changed -SourceIdentifier "${script:RunId}_FILE_CHANGED_$index"
                $script:Subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Deleted -SourceIdentifier "${script:RunId}_FILE_DELETED_$index"
                $script:Subscriptions += Register-ObjectEvent -InputObject $watcher -EventName Renamed -SourceIdentifier "${script:RunId}_FILE_RENAMED_$index"

                Write-Feed "WATCH" "Watching files under: $resolved" Blue
                $index++
            } else {
                Write-Feed "SKIP" "Path not found: $path" DarkYellow
            }
        }
    }

    if (-not $NoNetwork) {
        $script:SeenConnections = Get-TcpSnapshot
        Write-Feed "WATCH" "Watching TCP connection changes every $NetworkPollSeconds seconds" Blue
    }

    Write-Feed "READY" "Live feed running. Press Ctrl+C to stop." White

    while ($true) {
        $activityEvent = Wait-Event -Timeout 1

        if ($null -ne $activityEvent) {
            Handle-ActivityEvent -Event $activityEvent
            Remove-Event -EventIdentifier $activityEvent.EventIdentifier -ErrorAction SilentlyContinue

            while ($queuedEvent = Get-Event -ErrorAction SilentlyContinue | Select-Object -First 1) {
                Handle-ActivityEvent -Event $queuedEvent
                Remove-Event -EventIdentifier $queuedEvent.EventIdentifier -ErrorAction SilentlyContinue
            }
        }

        if (-not $NoNetwork) {
            $now = Get-Date
            if (($now - $script:LastNetworkPoll).TotalSeconds -ge $NetworkPollSeconds) {
                Poll-Network
                $script:LastNetworkPoll = $now
            }
        }
    }
}
catch {
    Write-Feed "ERROR" "$($_.Exception.Message)" Red
}
finally {
    Write-Feed "STOP" "Cleaning up watchers..." Yellow

    foreach ($subscription in $script:Subscriptions) {
        Unregister-Event -SubscriptionId $subscription.Id -ErrorAction SilentlyContinue
    }

    foreach ($watcher in $script:Watchers) {
        $watcher.EnableRaisingEvents = $false
        $watcher.Dispose()
    }

    Get-Event -ErrorAction SilentlyContinue | Where-Object {
        $_.SourceIdentifier -like "${script:RunId}*"
    } | Remove-Event -ErrorAction SilentlyContinue

    Write-Feed "DONE" "Terminal Activity Feed stopped." White
}
