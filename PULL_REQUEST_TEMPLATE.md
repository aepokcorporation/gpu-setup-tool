## Pull Request Checklist  

### Description  
Provide a brief description of the changes made in this pull request:  
- What problem does this PR solve?  
- What changes were made to address the issue?  

### Related Issues  
List any related issues:  
- Resolves #[Issue_Number]  
- Related to #[Other_Issue_Number]  

### Changes Made  
- [ ] Bug fix  
- [ ] New feature  
- [ ] Documentation update  
- [ ] Performance improvement  
- [ ] Refactor or code cleanup  

List the major changes:  
1. Updated `scripts/validate_gpu.py` to add functional TensorFlow tests.  
2. Improved CUDA installation progress feedback using `tqdm`.  

### Testing  
How were the changes tested? Provide clear instructions for reproducing the results:  
1. Ran `python3 scripts/setup_all.py` on an NVIDIA RTX 3090 system with Ubuntu 20.04.  
2. Validated the TensorFlow benchmark: `ResNet inference completed successfully`.  

### Checklist Before Submitting  
- [ ] Code has been tested and works as expected  
- [ ] All tests pass with no new warnings  
- [ ] Documentation has been updated (if applicable)  
- [ ] Related logs have been verified (validation_log.txt, install_log.txt)  

### Additional Notes  
(Optional) Add anything else reviewers should know, such as tradeoffs, assumptions, or future improvements:  
- Future support for AMD GPUs is planned in `v0.3.0`.  

---

## Contact  
If you have questions about this PR, you can reach me at [tyler@aepokcorp.com](mailto:tyler@aepokcorp.com).  
