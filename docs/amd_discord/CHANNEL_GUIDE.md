# Channel ID Quick Reference

Based on your screenshot, here are the key channels to export for Q6_K HIP graph optimization work.

## 🎯 Critical Channels (Must Export)

### 1. HIP Graphs Channel
```
From your screenshot, look for:
- Category: "HIP" or "Development"
- Channel: #hip-graphs or similar

Why critical:
- Direct discussions about HIP graph capture
- Solutions to graph capture errors (like our error 901)
- Code examples of graph-compatible kernels
- Known limitations and workarounds

Action:
1. Enable Developer Mode in Discord Settings
2. Right-click #hip-graphs → Copy ID
3. Paste into channels.conf as HIP_GRAPHS_ID
```

### 2. HIP General Discussion
```
Look for:
- Channel: #hip or #hip-api

Why important:
- General HIP programming discussions
- Kernel architecture patterns
- Performance optimization tips
- Common issues and solutions

Action:
Right-click #hip → Copy ID → Paste as HIP_ID
```

### 3. Help/Support
```
Look for:
- Channel: #help or #hip-help

Why valuable:
- Real-world problems and solutions
- Likely has graph capture issues discussed
- AMD engineer responses
- Workarounds for known bugs

Action:
Right-click #help → Copy ID → Paste as HELP_ID
```

## 📊 Important Channels (Should Export)

### 4. Performance Optimization
```
Look for:
- Channel: #optimization or #performance

Why useful:
- GPU kernel optimization techniques
- Memory access patterns
- Instruction-level optimizations
- Performance benchmarking discussions
```

### 5. GPU Programming
```
Look for:
- Channel: #gpu-programming or #cuda-to-hip

Why helpful:
- General GPU kernel patterns
- Device function best practices
- Memory management strategies
```

### 6. Bug Reports
```
Look for:
- Channel: #bug-reports or #issues

Why relevant:
- Known HIP graph bugs
- Official workarounds
- Fixed in version X discussions
```

## 🔧 Optional Channels (Nice to Have)

### Library-Specific
- `#rocBLAS` - If using BLAS operations
- `#rocFFT` - If using FFT operations
- `#rocRAND` - If using random number generation

### Documentation
- `#documentation` or `#docs` - Official docs links
- `#examples` - Code examples and tutorials
- `#announcement` - New releases and features

## 📝 How to Fill channels.conf

### Step 1: Open channels.conf
```bash
nano /home/feanor/Projects/rocmforge/docs/amd_discord/channels.conf
```

### Step 2: Replace Placeholders

Before:
```bash
HIP_GRAPHS_ID|hip-graphs|HIP Graph Capture Discussions
```

After (with real ID):
```bash
123456789012345678|hip-graphs|HIP Graph Capture Discussions
```

### Step 3: Minimum Configuration

For Q6_K work, configure at minimum:
```bash
123456789012345678|hip-graphs|HIP Graph Capture Discussions
234567890123456789|hip|HIP Runtime API Discussions
345678901234567890|help|General Help and Support
```

### Step 4: Run Export
```bash
cd /home/feanor/Projects/rocmforge/docs/amd_discord
./export_with_config.sh
```

## 🖼️ From Your Screenshot

Looking at the AMD server structure, prioritize:

1. **Any HIP category** (top priority)
   - Look for channels under "HIP" or "Development"
   - #hip-graphs is critical if it exists
   - #hip or #hip-api is next most important

2. **Support category** (high priority)
   - #help or #support
   - Likely contains graph capture troubleshooting

3. **Optimization category** (medium priority)
   - #performance or #optimization
   - Kernel optimization discussions

4. **Library categories** (low priority)
   - Only if relevant to your work
   - #rocBLAS, #rocFFT, etc.

## ⚡ Quick Workflow

```bash
# 1. Open Discord, enable Developer Mode
# Settings → Advanced → Toggle "Developer Mode"

# 2. Get channel IDs
# Right-click channel → Copy ID

# 3. Edit config
nano /home/feanor/Projects/rocmforge/docs/amd_discord/channels.conf

# 4. Export
cd /home/feanor/Projects/rocmforge/docs/amd_discord
./export_with_config.sh

# 5. Tell Claude
"I've exported Discord channels to /home/feanor/Projects/rocmforge/docs/amd_discord/exports/"
```

## 🎯 What We're Looking For

From these exports, Claude will extract:

1. **Graph Capture Patterns**
   - Device function vs inline code
   - Memory access strategies
   - Synchronization requirements

2. **Q6_K-Specific Info**
   - Any Q6_K discussions
   - Similar quantization schemes
   - Dequantization patterns

3. **Performance Insights**
   - Optimization techniques
   - Bottleneck identification
   - Best practices

4. **Known Issues**
   - Graph capture bugs
   - Workarounds and solutions
   - Version-specific issues

This will directly inform **Task #63: Refactor Q6_K kernel for HIP graph compatibility**

## ❓ Need Help Finding Channels?

If you can't find specific channels:
- Look through all categories in your screenshot
- Any HIP-related channel is valuable
- Export what you have access to
- More channels = more documentation!

## 🔒 Security Reminder

After getting channel IDs and exporting:
- Consider changing your Discord password
- This invalidates the token you used
- Keeps your account secure
