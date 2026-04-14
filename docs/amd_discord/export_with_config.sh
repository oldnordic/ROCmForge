#!/bin/bash

# Discord Channel Export Helper Script (with config file support)
# For rocmforge AMD Discord documentation project

set -e

EXPORT_DIR="/home/feanor/Projects/rocmforge/docs/amd_discord/exports"
CONFIG_FILE="/home/feanor/Projects/rocmforge/docs/amd_discord/channels.conf"
mkdir -p "$EXPORT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     AMD Discord Channel Export Helper                      ║"
echo "║     For rocmforge Q6_K HIP Graph Optimization             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo ""
    echo "Please edit channels.conf and add your channel IDs."
    echo "See channels.conf for instructions."
    exit 1
fi

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

# Read channels from config file
declare -a CHANNELS=()
declare -a CHANNEL_DESCRIPTIONS=()

while IFS='|' read -r channel_id dir_name description; do
    # Skip comments and empty lines
    [[ "$channel_id" =~ ^#.*$ ]] && continue
    [ -z "$channel_id" ] && continue

    # Skip placeholder IDs
    [[ "$channel_id" =~ .*_ID$ ]] && continue

    CHANNELS+=("$channel_id|$dir_name")
    CHANNEL_DESCRIPTIONS+=("$description")
done < "$CONFIG_FILE"

if [ ${#CHANNELS[@]} -eq 0 ]; then
    echo "❌ No valid channel IDs found in $CONFIG_FILE"
    echo ""
    echo "Please edit the file and replace CHANNEL_ID placeholders"
    echo "with actual Discord channel IDs (see instructions in file)."
    exit 1
fi

echo "Found ${#CHANNELS[@]} channels to export"
echo ""

# Export each channel
for i in "${!CHANNELS[@]}"; do
    IFS='|' read -r channel_id dir_name <<< "${CHANNELS[$i]}"
    description="${CHANNEL_DESCRIPTIONS[$i]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📥 Exporting: $description"
    echo "   Channel ID: $channel_id"
    echo "   Directory: $dir_name"
    echo ""

    # Create directory for this channel
    channel_dir="$EXPORT_DIR/$dir_name"
    mkdir -p "$channel_dir"

    # Export as plaintext (easier for Claude to parse)
    echo "  Exporting plaintext..."
    if discord-chat-exporter export \
        --token "$DISCORD_TOKEN" \
        --channel "$channel_id" \
        --format PlainText \
        --output "$channel_dir/messages.txt" \
        --after 2023-01-01 2>/dev/null; then
        echo "  ✅ Plaintext export complete"
    else
        echo "  ⚠️  Plaintext export failed (channel may not exist or no access)"
        continue
    fi

    # Also export as HTML for viewing
    echo "  Exporting HTML..."
    if discord-chat-exporter export \
        --token "$DISCORD_TOKEN" \
        --channel "$channel_id" \
        --format HtmlDark \
        --output "$channel_dir/messages.html" \
        --after 2023-01-01 2>/dev/null; then
        echo "  ✅ HTML export complete"
    else
        echo "  ⚠️  HTML export failed"
    fi

    # Create README for this channel
    cat > "$channel_dir/README.md" << EOF
# $description

Exported from AMD Developer Experience Discord server.

## Contents

- \`messages.txt\` - Plaintext export (for Claude analysis)
- \`messages.html\` - HTML export (for human viewing)

## Export Date

$(date)

## Channel Details

- **Channel ID:** $channel_id
- **Directory:** $dir_name
- **Description:** $description

## Export Stats

$(wc -l < "$channel_dir/messages.txt" 2>/dev/null | xargs echo "Total messages:" || echo "Total messages: 0")
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

for i in "${!CHANNELS[@]}"; do
    IFS='|' read -r channel_id dir_name <<< "${CHANNELS[$i]}"
    description="${CHANNEL_DESCRIPTIONS[$i]}"
    echo "- [$description]($dir_name/) - Channel ID: \`$channel_id\`" >> "$EXPORT_DIR/EXPORT_SUMMARY.md"
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

## Statistics

EOF

# Add message counts
total_messages=0
for i in "${!CHANNELS[@]}"; do
    IFS='|' read -r channel_id dir_name <<< "${CHANNELS[$i]}"
    description="${CHANNEL_DESCRIPTIONS[$i]}"
    msg_file="$EXPORT_DIR/$dir_name/messages.txt"

    if [ -f "$msg_file" ]; then
        count=$(wc -l < "$msg_file" 2>/dev/null || echo "0")
        total_messages=$((total_messages + count))
        echo "- **$description:** $count messages" >> "$EXPORT_DIR/EXPORT_SUMMARY.md"
    fi
done

echo "" >> "$EXPORT_DIR/EXPORT_SUMMARY.md"
echo "**Total Messages Across All Channels:** $total_messages" >> "$EXPORT_DIR/EXPORT_SUMMARY.md"

cat >> "$EXPORT_DIR/EXPORT_SUMMARY.md" << EOF

## Next Steps

1. Review the exports in \`exports/\` directories
2. Tell Claude where the exports are located
3. Claude will analyze and organize the information
4. Key findings will be added to project documentation
5. Apply insights to Q6_K kernel refactoring (Task #63)

## Configuration

Exported using configuration from:
\`../channels.conf\`

To add more channels:
1. Edit \`../channels.conf\`
2. Add channel ID, directory name, and description
3. Re-run this script
EOF

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Export Complete!                                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Export directory: $EXPORT_DIR"
echo "📄 Summary: $EXPORT_DIR/EXPORT_SUMMARY.md"
echo ""
echo "📊 Exported ${#CHANNELS[@]} channels"
echo ""
echo "Next steps:"
echo "  1. Review the exports in: $EXPORT_DIR"
echo "  2. Tell Claude: I've exported Discord channels to $EXPORT_DIR"
echo "  3. Claude will analyze and extract key information"
echo ""
echo "⚠️  Security reminder: Consider changing your Discord password"
echo "   to invalidate the token you just used."
echo ""
