//
//  LNKMemoryBufferManager.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

typedef struct _LNKMemoryBufferManager LNKMemoryBufferManager;
typedef LNKMemoryBufferManager *LNKMemoryBufferManagerRef;

LNKMemoryBufferManagerRef LNKMemoryBufferManagerCreate();
void LNKMemoryBufferManagerFree(LNKMemoryBufferManagerRef manager);

LNKFloat *LNKMemoryBufferManagerAllocBlock(LNKMemoryBufferManagerRef manager, LNKSize size);
void LNKMemoryBufferManagerFreeBlock(LNKMemoryBufferManagerRef manager, LNKFloat *buffer, LNKSize size);

LNKMemoryBufferManagerRef LNKGetCurrentMemoryBufferManager();
