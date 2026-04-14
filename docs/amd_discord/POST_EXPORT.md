# After Exporting Discord Channels

Once you've exported AMD Discord channels, follow this guide to process the data with Claude.

## Quick Start

1. **Verify exports exist:**
   ```bash
   ls -la /home/feanor/Projects/rocmforge/docs/amd_discord/exports/
   ```

2. **Tell Claude about the exports:**
   ```
   I've exported Discord channels to /home/feanor/Projects/rocmforge/docs/amd_discord/exports/
   ```

3. **Claude will:**
   - Parse all plaintext exports
   - Extract documentation links, code snippets, technical discussions
   - Organize information into structured documents
   - Cross-reference with Q6_K HIP graph work
   - Identify actionable insights for optimization

## What Claude Will Extract

### 1. Documentation Links
- Official ROCm/HIP documentation URLs
- AMD GPUOpen resources
- GitHub repositories with examples
- Performance tuning guides

### 2. Code Snippets
- HIP kernel examples
- Graph capture patterns
- Memory optimization techniques
- Device function implementations

### 3. Technical Discussions
- Common issues and solutions
- Performance bottlenecks
- Graph compatibility requirements
- Known bugs and workarounds

### 4. Developer Insights
- AMD engineer recommendations
- Best practices
- Performance tips
- Architecture patterns

## Output Structure

Claude will create organized documentation:

```
docs/amd_discord/
├── exports/                      # Raw exports (your data)
│   ├── hip-graphs/
│   ├── roc-documentation/
│   └── performance/
├── extracted/                    # Processed information (Claude creates)
│   ├── documentation_links.md    # All external doc links
│   ├── code_examples.md          # Code snippets organized by topic
│   ├── graph_capture_guide.md    # HIP graph best practices
│   ├── known_issues.md           # Bugs and workarounds
│   └── performance_tips.md       # Optimization recommendations
└── analysis/                     # Applied to Q6_K work
    ├── q6_k_refactoring_insights.md  # Specific to Q6_K
    ├── graph_compatibility_check.md   # What makes kernels work
    └── recommended_changes.md         # Actionable improvements
```

## Example Claude Prompts

### Analyze All Exports
```
Parse the Discord exports in /home/feanor/Projects/rocmforge/docs/amd_discord/exports/ and extract:
1. All HIP graph documentation links
2. Code examples showing graph-compatible kernels
3. Discussions about Q6_K or similar quantization
4. Performance optimization tips for HIP kernels
5. Known issues with graph capture

Organize everything into structured markdown files in the extracted/ directory.
```

### Focus on Q6_K
```
Focus the analysis on information relevant to Q6_K quantization:
- Graph compatibility requirements
- Device function patterns
- Memory access optimization
- Bit manipulation techniques
- Similar quantization schemes (Q4_K, Q8_K)

Create specific recommendations for refactoring our Q6_K kernel.
```

### Extract Code Examples
```
Extract all HIP kernel code examples from the exports.
For each example:
- Document what it demonstrates
- Explain why it's graph-compatible (or not)
- Note any patterns we should follow for Q6_K
- Compare with our current implementations
```

## Security and Privacy

### What to Keep Private
- ❌ Don't share raw exports outside this project
- ❌ Don't commit sensitive developer discussions
- ❌ Don't redistribute private channel content
- ❌ Don't quote AMD developers without attribution

### What's Safe to Share
- ✅ Summarized technical information
- ✅ Public documentation links
- ✅ General best practices
- ✅ Code examples (if not marked confidential)

### Git Best Practices

The `exports/` directory contains raw Discord data:
```bash
# Add to .gitignore
echo "docs/amd_discord/exports/" >> /home/feanor/Projects/rocmforge/.gitignore
```

The `extracted/` directory contains processed information:
```bash
# Safe to commit (summarized, no private content)
git add docs/amd_discord/extracted/
git commit -m "docs: add extracted AMD Discord documentation"
```

## Integration with Q6_K Work

After Claude processes the exports, we'll have:

1. **Graph-Compatible Patterns**
   - Device function examples
   - Memory access strategies
   - Kernel architecture recommendations

2. **Q6_K-Specific Insights**
   - If Q6_K was discussed
   - Similar quantization work
   - Performance benchmarks

3. **Official AMD Guidance**
   - Graph capture requirements
   - Known limitations
   - Recommended practices

4. **Refactoring Strategy**
   - Specific code patterns to follow
   - Common pitfalls to avoid
   - Performance optimization opportunities

This will directly inform **Task #63: Refactor Q6_K kernel for HIP graph compatibility**

## Troubleshooting

### Claude Can't Find Exports
- Verify the path is correct
- Check file permissions
- Ensure plaintext (.txt) exports exist

### Extraction Misses Information
- Specify what to look for
- Point to specific channels or messages
- Provide context about what you need

### Too Much Information
- Ask Claude to focus on specific topics
- Limit to certain channels
- Request summaries instead of full extraction

## Next Steps

1. ✅ Export Discord channels using `export.sh`
2. ✅ Tell Claude where exports are located
3. ✅ Claude processes and extracts information
4. ⏳ Review extracted documentation
5. ⏳ Apply insights to Q6_K refactoring (Task #63)
6. ⏳ Test refactored kernel with graph capture

## Questions?

If you need help:
- Review `SETUP.md` for installation issues
- Review `USAGE.md` for export problems
- Ask Claude specific questions about the exported content

---

**Remember:** The goal is to learn from AMD's official guidance to make Q6_K graph-compatible and close the 3.9x performance gap to Q4_K.
