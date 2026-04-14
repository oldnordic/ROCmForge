# Quick Start: Export AMD Discord for Q6_K Work

This is the fastest path from zero to having AMD documentation extracted for our Q6_K HIP graph optimization.

## Prerequisites Checklist

- [ ] You're a member of AMD Developer Experience Discord server
- [ ] You have access to HIP/ROCm documentation channels
- [ ] 5-10 minutes of time

## Step-by-Step (5 Minutes)

### 1. Install DiscordChatExporter (2 minutes)

```bash
# Install .NET SDK
yay -S dotnet-sdk

# Install DiscordChatExporter
dotnet tool install --global DiscordChatExporter

# Add to PATH (for fish shell)
echo 'set -gx PATH $HOME/.dotnet/tools $PATH' >> ~/.config/fish/config.fish
source ~/.config/fish/config.fish
```

### 2. Get Your Discord Token (1 minute)

1. Open Discord in browser: https://discord.com
2. Press `Ctrl+Shift+I` (open DevTools)
3. Go to "Application" tab → "Local Storage" → "https://discord.com"
4. Find "token" and copy its value
5. ⚠️ Keep this token private!

### 3. Find Channel IDs (1 minute)

1. In Discord Settings → Advanced → Enable "Developer Mode"
2. Right-click on HIP documentation channel
3. Select "Copy ID"
4. Note: This is the numeric ID, not the channel name

### 4. Run Export Script (1 minute)

```bash
cd /home/feanor/Projects/rocmforge/docs/amd_discord
./export.sh
```

When prompted:
- Paste your Discord token
- Wait for exports to complete

## That's It!

Your exports are now in:
```
/home/feanor/Projects/rocmforge/docs/amd_discord/exports/
```

## Tell Claude

```
I've exported Discord channels to /home/feanor/Projects/rocmforge/docs/amd_discord/exports/
```

Claude will:
- ✅ Parse all exports
- ✅ Extract documentation links
- ✅ Identify code examples
- ✅ Summarize technical discussions
- ✅ Create actionable recommendations for Q6_K

## What You'll Get

Organized documentation in `extracted/`:
- HIP graph best practices
- Device function patterns
- Performance optimization tips
- Known issues and workarounds
- Q6_K-specific insights

This directly supports **Task #63: Refactor Q6_K kernel for HIP graph compatibility**

## Security Reminder

After exporting, consider changing your Discord password to invalidate the token you used.

## Need More Details?

- `SETUP.md` - Full installation guide
- `USAGE.md` - Detailed usage instructions
- `POST_EXPORT.md` - What happens after export
- `README.md` - Project overview

---

**Goal:** Extract AMD's official HIP graph guidance to make Q6_K graph-compatible and achieve 2.2-3.7x performance improvement.
