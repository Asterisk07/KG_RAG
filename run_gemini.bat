@echo off
@REM to run press : .\run_gemini.bat
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-1.5-flash --mode 1 --num 2
@REM python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-1.5-flash
@REM python -m kg_rag.rag_based_generation.GPT.run_mcq_qa gemini-1.5-flash --num 1 --eval