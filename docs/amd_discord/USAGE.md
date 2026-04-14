# DiscordChatExporter Usage Guide

This guide shows you how to export AMD Developer Experience Discord channels for documentation analysis.

## Important: Account Token (NOT Bot Token)

You'll need your **Discord user token** (not a bot token) to export channels you have access to.

### How to Get Your Discord Token

⚠️ **Keep this token private!** Never share it with anyone.

1. Open Discord in your browser (discord.com)
2. Press `Ctrl+Shift+I` (Linux/Windows) or `Cmd+Option+I` (Mac) to open Developer Tools
3. Go to the "Application" tab (Chrome) or "Storage" tab (Firefox)
4. Expand "Local Storage" on the left
5. Click on "https://discord.com"
6. Find "token" in the list
7. Copy the value (it's a long string like "MTE...")

**Security Note:** This token gives full access to your account. After exporting, consider changing your password (which invalidates the token).

## Exporting Channels

### Option 1: Command Line (CLI)

```bash
# Basic export (all messages from a channel)
discord-chat-exporter export \
  --token "YOUR_TOKEN_HERE" \
  --channel "CHANNEL_ID" \
  --output /home/feanor/Projects/rocmforge/docs/amd_discord/exports/channel-name.html

# Export as plaintext (easier for Claude to parse)
discord-chat-exporter export \
  --token "YOUR_TOKEN_HERE" \
  --channel "CHANNEL_ID" \
  --format PlainText \
  --output /home/feanor/Projects/rocmforge/docs/amd_discord/exports/channel-name.txt

# Export with date range
discord-chat-exporter export \
  --token "YOUR_TOKEN_HERE" \
  --channel "CHANNEL_ID" \
  --after 2024-01-01 \
  --before 2025-12-31 \
  --output /home/feanor/Projects/rocmforge/docs/amd_discord/exports/channel-name.html
```

### Option 2: Interactive CLI

```bash
# Run without arguments for interactive mode
discord-chat-exporter export

# It will prompt for:
# - Token
# - Channel selection (from list)
# - Output format
# - Output path
```

### Option 3: GUI

If you installed the GUI version:

```bash
~/Applications/DiscordChatExporter/DiscordChatExporter.Gui
```

## Finding Channel IDs

You need the **numeric channel ID**, not the name.

### Method 1: Enable Developer Mode in Discord

1. Open Discord Settings
2. Go to "Advanced"
3. Enable "Developer Mode"
4. Right-click on any channel
5. Select "Copy ID" to get the channel ID

### Method 2: From URL

1. Open the channel in Discord web or desktop app
2. The URL will be like: `https://discord.com/channels/SERVER_ID/CHANNEL_ID`
3. Copy the last number (CHANNEL_ID)

## Recommended AMD Channels to Export

Based on our Q6_K HIP graph work, these channels are most valuable:

### Core HIP/ROCm Documentation
- `#hip-graphs` - HIP graph capture discussions
- `#roc-documentation` - Official ROCm documentation links
- `#hip-examples` - Code examples and best practices
- `#performance-optimization` - GPU performance tuning

### Problem-Solving
- `#hip-help` - Common issues and solutions
- `#bug-reports` - Known bugs and workarounds
- `#graph-capture-issues` - Specific to graph capture problems

### Development Discussion
- `#kernel-development` - GPU kernel programming
- `#quantization` - Model quantization techniques
- `#memory-management` - GPU memory optimization

## Export Workflow

```bash
# 1. Create export directory
mkdir -p /home/feanor/Projects/rocmforge/docs/amd_discord/exports/hip-graphs

# 2. Export the channel (plaintext for easier parsing)
discord-chat-exporter export \
  --token "YOUR_TOKEN" \
  --channel "CHANNEL_ID" \
  --format PlainText \
  --output /home/feanor/Projects/rocmforge/docs/amd_discord/exports/hip-graphs/hip-graphs.txt

# 3. Repeat for other channels
```

## Helper Script

I've created `export.sh` to simplify the process:

```bash
cd /home/feanor/Projects/rocmforge/docs/amd_discord
./export.sh
```

This will:
1. Prompt for your Discord token
2. List recommended channels
3. Export each channel to organized directories
4. Generate a summary of what was exported

## After Exporting

Once you've exported channels:

1. **Tell Claude where the exports are:**
   ```
   I've exported Discord channels to /home/feanor/Projects/rocmforge/docs/amd_discord/exports/
   ```

2. **Claude will:**
   - Parse all exported files
   - Extract documentation links
   - Identify code snippets
   - Summarize technical discussions
   - Cross-reference with Q6_K work
   - Create organized documentation

## Security Best Practices

1. **Never commit your token** to git
2. **Don't share exported logs** publicly (private discussions)
3. **Change password after exporting** (invalidates token)
4. **Store exports in private directories**
5. **Only export channels you have permission to read**

## Troubleshooting

### "Invalid Token"

- Token might be expired (regenerate from Discord Dev Tools)
- Make sure you copied the entire token (no spaces)

### "Missing Access"

- You don't have permission to read that channel
- Only export channels you're a member of

### "Rate Limited"

Discord has rate limits:
- Wait a few minutes between exports
- Export fewer channels
- Export smaller date ranges

## Next Steps

After exporting, see `POST_EXPORT.md` for how to process the exported data with Claude.
