# @app.post("/query", response_model=QueryResponse)
# async def query_assets(request: QueryRequest, db: Session = Depends(get_db)):
#     question = request.question
#     print(f"Question: {question}")
    
#     # Direct lookup approach for asset tags in the question
#     # Assuming asset tags are in UUID format
#     import re
#     uuid_pattern = r'[-1-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
#     found_tags = re.findall(uuid_pattern, question)
    
#     if found_tags:
#         asset_tag = found_tags[-1]
#         print(f"Found asset tag in question: {asset_tag}")
#     else:
#         # If no direct UUID found, then try the QA approach
#         context = " ".join([f"Asset: {asset.name}, Tag: {asset.asset_tag}, Serial: {asset.serial}" 
#                            for asset in db.query(Asset).all()])
#         print(f"Context length: {len(context)}")
        
#         nlp_input = {"question": question, "context": context}
#         result = app.state.qa_pipeline(nlp_input)
#         print(f"QA Pipeline Result: {result}")
        
#         # Try to extract a tag-like pattern from the answer
#         answer_tags = re.findall(uuid_pattern, result['answer'])
#         if answer_tags:
#             asset_tag = answer_tags[-1]
#         else:
#             asset_tag = result['answer']
    
#     # Look up the asset
#     asset = db.query(Asset).filter(Asset.asset_tag == asset_tag).first()
#     if not asset:
#         return QueryResponse(answer=f"Asset not found for tag: {asset_tag}")

#     answer = f"Asset Name: {asset.name}, Asset Tag: {asset.asset_tag}, Serial: {asset.serial}, Purchase Date: {asset.purchase_date}"
#     return QueryResponse(answer=answer)
