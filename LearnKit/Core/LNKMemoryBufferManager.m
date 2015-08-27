//
//  LNKMemoryBufferManager.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMemoryBufferManager.h"

#import <pthread.h>

#define FIRST_PAGE_LENGTH	4096
#define TOTAL_PAGES			1048575 // floor(UINT32_MAX / FIRST_PAGE_LENGTH)
#define MAX_SUPPORTED_SIZE	4294963200

#define FIXED_BUCKET_DIM	8

struct _LNKMemoryBufferBucket {
	LNKSize size;
	LNKFloat *buffer;
	
	struct _LNKMemoryBufferBucket *previous;
	struct _LNKMemoryBufferBucket *next;
};

struct _LNKMemoryBufferManager {
	struct _LNKMemoryBufferBucket **pages;
	struct _LNKMemoryBufferBucket *head;
};

LNKMemoryBufferManagerRef LNKMemoryBufferManagerCreate() {
	LNKMemoryBufferManagerRef manager = malloc(sizeof(LNKMemoryBufferManager));
	manager->head = NULL;
	manager->pages = calloc(sizeof(struct _LNKMemoryBufferBucket *), TOTAL_PAGES);
	
	return manager;
}

void LNKMemoryBufferManagerFree(LNKMemoryBufferManagerRef manager) {
	NSCAssert(manager, @"The manager must not be NULL");
	
	struct _LNKMemoryBufferBucket **pages = manager->pages;
	struct _LNKMemoryBufferBucket *bucket = manager->head;
	
	while (bucket) {
		struct _LNKMemoryBufferBucket *next = bucket->next;
		free(bucket);
		bucket = next;
	}
	
	free(pages);
	free(manager);
}

LNKFloat *LNKMemoryBufferManagerAllocBlock(LNKMemoryBufferManagerRef manager, LNKSize size) {
	return malloc(size * sizeof(LNKFloat));
	
	NSCAssert(manager, @"The manager must not be NULL");
	NSCAssert(size, @"The size must be greater than 0");
	
	// We don't support 64-bit sizes.
	if (size > MAX_SUPPORTED_SIZE)
		return malloc(size * sizeof(LNKFloat));
	
	LNKSize firstPageOffset = size / FIRST_PAGE_LENGTH;
	struct _LNKMemoryBufferBucket *buckets = manager->pages[firstPageOffset];
	
	if (!buckets)
		return malloc(size * sizeof(LNKFloat));
	
	for (LNKSize i = 0; i < FIXED_BUCKET_DIM; i++) {
		struct _LNKMemoryBufferBucket *bucket = buckets + i;
		
		if (bucket->size == size) {
			struct _LNKMemoryBufferBucket *previous = bucket->previous;
			struct _LNKMemoryBufferBucket *next = bucket->next;
			
			if (previous == NULL) {
//				assert(manager->head == bucket);
				manager->head = next;
			}
			
			if (previous) previous->next = next;
			if (next) next->previous = previous;
			
			bucket->size = 0;
			return bucket->buffer;
		}
	}
	
	return malloc(size * sizeof(LNKFloat));
}

void LNKMemoryBufferManagerFreeBlock(LNKMemoryBufferManagerRef manager, LNKFloat *buffer, LNKSize size) {
	free(buffer);
	return;
	
	NSCAssert(manager, @"The manager must not be NULL");
	NSCAssert(buffer, @"The buffer must not be NULL");
	NSCAssert(size, @"The size must be greater than 0");
	
	// We don't support 64-bit sizes.
	if (size > MAX_SUPPORTED_SIZE)
		free(buffer);
	
	LNKSize firstPageOffset = size / FIRST_PAGE_LENGTH;
	struct _LNKMemoryBufferBucket *buckets = manager->pages[firstPageOffset];
	struct _LNKMemoryBufferBucket *freeBucket = NULL;
	
	if (!buckets) {
		manager->pages[firstPageOffset] = calloc(sizeof(struct _LNKMemoryBufferBucket), FIXED_BUCKET_DIM);
		freeBucket = manager->pages[firstPageOffset];
	}
	else {
		for (LNKSize i = 0; i < FIXED_BUCKET_DIM; i++) {
			struct _LNKMemoryBufferBucket *bucket = buckets + i;
			
			if (bucket->size == 0) {
				freeBucket = bucket;
				break;
			}
		}
	}
	
	if (!freeBucket) {
		free(buffer);
		return;
	}
	
	freeBucket->buffer = buffer;
	freeBucket->size = size;
	freeBucket->previous = NULL;
	freeBucket->next = manager->head;
	manager->head = freeBucket;
}


static pthread_key_t gMemoryBufferKey;

static void destroyMemoryBuffer(void *buffer) {
	LNKMemoryBufferManagerFree((LNKMemoryBufferManagerRef)buffer);
}

LNKMemoryBufferManagerRef LNKGetCurrentMemoryBufferManager() {
	static dispatch_once_t onceToken;
	dispatch_once(&onceToken, ^{
		pthread_key_create(&gMemoryBufferKey, &destroyMemoryBuffer);
	});
	
	LNKMemoryBufferManagerRef memoryManager = pthread_getspecific(gMemoryBufferKey);
	
	if (!memoryManager) {
		memoryManager = LNKMemoryBufferManagerCreate();
		pthread_setspecific(gMemoryBufferKey, memoryManager);
	}
	
	return memoryManager;
}
