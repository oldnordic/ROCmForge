# AMD Discord Documentation Export

This directory contains exported documentation and discussions from the AMD Developer Experience Discord server.

## Purpose

To capture AMD's official HIP/ROCm documentation, discussions, and code examples shared by AMD developers for use in the rocmforge Q6_K optimization project.

## Setup

See `SETUP.md` for installation instructions for DiscordChatExporter.

## Exporting Channels

See `USAGE.md` for step-by-step guide on how to export Discord channels.

## File Structure

```
amd_discord/
├── README.md                 # This file
├── SETUP.md                  # Installation instructions
├── USAGE.md                  # How to export channels
├── export.sh                 # Helper script for exports
└── exports/                  # Exported channel data
    ├── hip-graphs/           # HIP graph discussions
    ├── roc-documentation/    # ROCm documentation links
    ├── optimization/         # Performance optimization tips
    └── bug-reports/          # Known issues and workarounds
```

## Important Notes

⚠️ **Account Safety:**
- Never share your Discord token
- Only export channels you have permission to read
- Use official DiscordChatExporter tool (not unofficial scrapers)

⚠️ **Terms of Service:**
- Only export for personal archival/analysis
- Don't redistribute private discussions
- Respect AMD developers' privacy

## Contributing

When exporting new channels:
1. Use meaningful directory names
2. Include a README in each export folder describing the channel
3. Update this README with what was exported
4. Run the `parse_exports.sh` script to extract key information

## Next Steps After Export

Once you've exported channels, tell Claude:
> "I've exported Discord channels to `/home/feanor/Projects/rocmforge/docs/amd_discord/exports/<channel-name>`"

Claude will then:
1. Parse the exported files
2. Extract documentation links, code snippets, and technical discussions
3. Organize information into structured markdown documents
4. Cross-reference with existing Q6_K optimization work
5. Identify key insights for making Q6_K graph-compatible
