//
//  LNKMemoryBufferManager.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

typedef struct _LNKMemoryBufferManager LNKMemoryBufferManager;
typedef LNKMemoryBufferManager *LNKMemoryBufferManagerRef;

extern LNKMemoryBufferManagerRef LNKMemoryBufferManagerCreate();
extern void LNKMemoryBufferManagerFree(LNKMemoryBufferManagerRef manager);

extern LNKFloat *LNKMemoryBufferManagerAllocBlock(LNKMemoryBufferManagerRef manager, LNKSize size);
extern void LNKMemoryBufferManagerFreeBlock(LNKMemoryBufferManagerRef manager, LNKFloat *buffer, LNKSize size);

extern LNKMemoryBufferManagerRef LNKGetCurrentMemoryBufferManager();
