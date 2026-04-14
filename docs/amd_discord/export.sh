#!/bin/bash

# Discord Channel Export Helper Script
# For rocmforge AMD Discord documentation project

set -e

EXPORT_DIR="/home/feanor/Projects/rocmforge/docs/amd_discord/exports"
mkdir -p "$EXPORT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     AMD Discord Channel Export Helper                      ║"
echo "║     For rocmforge Q6_K HIP Graph Optimization             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if discord-chat-exporter is installed
if ! command -v discord-chat-exporter &> /dev/null; then
    echo "❌ DiscordChatExporter not found!"
    echo ""
    echo "Please install it first:"
    echo "  yay -S dotnet-sdk"
    echo "  dotnet tool install --global DiscordChatExporter"
    echo "  source ~/.config/fish/config.fish"
    echo ""
    exit 1
fi

# Prompt for token
echo "Enter your Discord token (will be hidden):"
read -s DISCORD_TOKEN
echo ""

if [ -z "$DISCORD_TOKEN" ]; then
    echo "❌ Token cannot be empty!"
    exit 1
fi

# Define recommended channels
# Format: "CHANNEL_ID|Directory Name|Description"
declare -a CHANNELS=(
    # You'll need to replace these with actual channel IDs from AMD Discord
    # Right-click channel → Copy ID (with Developer Mode enabled)
    "YOUR_CHANNEL_ID_HERE|hip-graphs|HIP Graph Capture Discussions"
    "YOUR_CHANNEL_ID_HERE|roc-documentation|ROCm Official Documentation"
    "YOUR_CHANNEL_ID_HERE|performance|GPU Performance Optimization"
    "YOUR_CHANNEL_ID_HERE|help|HIP Help and Support"
)

echo "Found ${#CHANNELS[@]} channels to export"
echo ""

# Export each channel
for channel in "${CHANNELS[@]}"; do
    IFS='|' read -r channel_id dir_name description <<< "$channel"

    if [ "$channel_id" = "YOUR_CHANNEL_ID_HERE" ]; then
        echo "⚠️  Skipping $dir_name (needs actual channel ID)"
        echo ""
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📥 Exporting: $description"
    echo "   Directory: $dir_name"
    echo ""

    # Create directory for this channel
    channel_dir="$EXPORT_DIR/$dir_name"
    mkdir -p "$channel_dir"

    # Export as plaintext (easier for Claude to parse)
    discord-chat-exporter export \
        --token "$DISCORD_TOKEN" \
        --channel "$channel_id" \
        --format PlainText \
        --output "$channel_dir/messages.txt" \
        --after 2023-01-01

    # Also export as HTML for viewing
    discord-chat-exporter export \
        --token "$DISCORD_TOKEN" \
        --channel "$channel_id" \
        --format HtmlDark \
        --output "$channel_dir/messages.html" \
        --after 2023-01-01

    # Create README for this channel
    cat > "$channel_dir/README.md" << EOF
# $description

Exported from AMD Developer Experience Discord server.

## Contents

- \`messages.txt\` - Plaintext export (for Claude analysis)
- \`messages.html\` - HTML export (for human viewing)

## Export Date

$(date)

## Channel ID

$channel_id
EOF

    echo "✅ Exported to: $channel_dir"
    echo ""
done

# Generate summary
cat > "$EXPORT_DIR/EXPORT_SUMMARY.md" << EOF
# AMD Discord Export Summary

**Export Date:** $(date)
**Total Channels:** ${#CHANNELS[@]}

## Exported Channels

EOF

for channel in "${CHANNELS[@]}"; do
    IFS='|' read -r channel_id dir_name description <<< "$channel"
    if [ "$channel_id" != "YOUR_CHANNEL_ID_HERE" ]; then
        echo "- [$description]($dir_name/) - Channel ID: $channel_id" >> "$EXPORT_DIR/EXPORT_SUMMARY.md"
    fi
done

cat >> "$EXPORT_DIR/EXPORT_SUMMARY.md" << EOF

## Usage

Tell Claude:
> "I've exported Discord channels to \`/home/feanor/Projects/rocmforge/docs/amd_discord/exports/\`"

Claude will parse all exports and extract:
- Documentation links
- Code snippets
- Technical discussions
- Bug reports and workarounds
- Performance optimization tips

## Next Steps

1. Review the exports in \`exports/\` directories
2. Tell Claude where the exports are located
3. Claude will analyze and organize the information
4. Key findings will be added to project documentation
EOF

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Export Complete!                                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Export directory: $EXPORT_DIR"
echo "📄 Summary: $EXPORT_DIR/EXPORT_SUMMARY.md"
echo ""
echo "Next steps:"
echo "  1. Review the exports in: $EXPORT_DIR"
echo "  2. Tell Claude: I've exported Discord channels to $EXPORT_DIR"
echo "  3. Claude will analyze and extract key information"
echo ""
echo "⚠️  Security reminder: Consider changing your Discord password"
echo "   to invalidate the token you just used."
echo ""
