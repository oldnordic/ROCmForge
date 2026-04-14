# DiscordChatExporter Setup Guide

This guide will help you install DiscordChatExporter on your Arch Linux system.

## Prerequisites

DiscordChatExporter requires .NET 6.0 SDK or later.

## Installation Steps

### Step 1: Install .NET SDK

```bash
# Install .NET SDK using yay (AUR helper)
yay -S dotnet-sdk

# Or using pacman directly
sudo pacman -S dotnet-sdk
```

### Step 2: Verify Installation

```bash
# Check .NET version
dotnet --version

# Should show: 6.0.x, 7.0.x, or 8.0.x
```

### Step 3: Install DiscordChatExporter

```bash
# Install as a global .NET tool
dotnet tool install --global DiscordChatExporter

# Add .NET tools to your PATH (add to ~/.config/fish/config.fish)
# set -gx PATH $HOME/.dotnet/tools $PATH
```

### Step 4: Verify Installation

```bash
# Check if DiscordChatExporter is available
discord-chat-exporter --version

# If command not found, reload your shell:
source ~/.config/fish/config.fish

# Or use full path:
~/.dotnet/tools/discord-chat-exporter --version
```

## Alternative: Direct Download

If the .NET tool doesn't work, you can download the standalone binary:

```bash
# Create directory for the tool
mkdir -p ~/Applications
cd ~/Applications

# Download latest Linux release (check for newer version at GitHub)
wget https://github.com/Tyrrrz/DiscordChatExporter/releases/download/2.42.0/DiscordChatExporter-linux-gui-x64.zip

# Unzip
unzip DiscordChatExporter-linux-gui-x64.zip -d DiscordChatExporter

# Run
cd DiscordChatExporter
./DiscordChatExporter.Cli
```

## Troubleshooting

### "dotnet: command not found"

Install .NET SDK:
```bash
yay -S dotnet-sdk
```

### "discord-chat-exporter: command not found"

Add .NET tools to PATH:
```bash
# For fish shell (which you use)
echo 'set -gx PATH $HOME/.dotnet/tools $PATH' >> ~/.config/fish/config.fish
source ~/.config/fish/config.fish
```

### Permission Denied

Make the binary executable:
```bash
chmod +x ~/Applications/DiscordChatExporter/DiscordChatExporter.Cli
```

## Next Steps

After installation, see `USAGE.md` for how to export Discord channels.

## References

- DiscordChatExporter GitHub: https://github.com/Tyrrrz/DiscordChatExporter
- Official Documentation: https://github.com/Tyrrrz/DiscordChatExporter/blob/master/.wiki/Usage.md
