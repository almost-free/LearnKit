//
//  LNKFastArray.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKFastArray.h"

#define DEFAULT_COUNT 128

struct _LNKFastArray {
	LNKSize count;
	LNKSize capacity;
	LNKSize elementWidth;
	void *buffer;
};

LNKFastArrayRef LNKFastArrayCreate(LNKSize elementWidth) {
	assert(elementWidth);
	
	LNKFastArrayRef array = malloc(sizeof(LNKFastArray));
	array->capacity = DEFAULT_COUNT;
	array->count = 0;
	array->elementWidth = elementWidth;
	array->buffer = malloc(elementWidth * DEFAULT_COUNT);
	return array;
}

void LNKFastArrayFree(LNKFastArrayRef array) {
	assert(array);
	
	free(array->buffer);
	free(array);
}

void LNKFastArrayAddElement(LNKFastArrayRef array, void *bytes) {
	assert(array);
	assert(bytes);
	
	if (array->count == array->capacity) {
		array->capacity *= 2;
		array->buffer = realloc(array->buffer, array->capacity * array->elementWidth);
	}
	
	char *buffer = (char *)array->buffer;
	memcpy(&buffer[array->count * array->elementWidth], bytes, array->elementWidth);
	array->count++;
}

void *LNKFastArrayElementAtIndex(LNKFastArrayRef array, LNKSize index) {
	assert(array);
	assert(index < array->count);
	
	char *bytes = (char *)array->buffer;
	return &bytes[index * array->elementWidth];
}

LNKSize LNKFastArrayElementCount(LNKFastArrayRef array) {
	assert(array);
	
	return array->count;
}
