# Clears stale Terminal Activity Feed event subscribers from the current PowerShell session.
# Use this only if a previous run was interrupted and a new run reports duplicate subscribers.

$patterns = @(
    "TAF_*",
    "PROC_START",
    "PROC_STOP",
    "FILE_CREATED_*",
    "FILE_CHANGED_*",
    "FILE_DELETED_*",
    "FILE_RENAMED_*"
)

function Test-ActivityFeedSourceId {
    param([string]$SourceIdentifier)

    foreach ($pattern in $patterns) {
        if ($SourceIdentifier -like $pattern) {
            return $true
        }
    }

    return $false
}

Get-EventSubscriber -ErrorAction SilentlyContinue | Where-Object {
    Test-ActivityFeedSourceId -SourceIdentifier $_.SourceIdentifier
} | ForEach-Object {
    Unregister-Event -SubscriptionId $_.SubscriptionId -ErrorAction SilentlyContinue
}

Get-Event -ErrorAction SilentlyContinue | Where-Object {
    Test-ActivityFeedSourceId -SourceIdentifier $_.SourceIdentifier
} | Remove-Event -ErrorAction SilentlyContinue

Write-Host "Terminal Activity Feed event cleanup complete." -ForegroundColor Green
