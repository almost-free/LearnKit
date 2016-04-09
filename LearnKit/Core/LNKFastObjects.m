//
//  LNKFastObjects.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKFastObjects.h"

#define DEFAULT_COUNT 128

struct _LNKFastObject {
	LNKSize retainCount;
	void *buffer;
};

struct _LNKFastArray {
	struct _LNKFastObject parent;
	LNKSize count;
	LNKSize capacity;
};

struct _LNKFastString {
	struct _LNKFastObject parent;
};

void LNKFastObjectRetain(LNKFastObjectRef object) {
	NSCAssert(object != NULL, @"The object must not be NULL");
	object->retainCount++;
}

void LNKFastObjectRelease(LNKFastObjectRef object) {
	NSCAssert(object != NULL, @"The object must not be NULL");

	object->retainCount--;
	NSCAssert(object->retainCount >= 0, @"Imbalanced retain/release");

	if (object->retainCount == 0) {
		if (object->buffer) {
			free(object->buffer);
			object->buffer = NULL;
		}

		free(object);
	}
}

LNKFastArrayRef LNKFastArrayCreate() {
	LNKFastArrayRef array = calloc(sizeof(LNKFastArray), 1);
	array->capacity = DEFAULT_COUNT;
	array->parent.retainCount = 1;
	array->parent.buffer = malloc(sizeof(LNKFastObjectRef) * DEFAULT_COUNT);
	return array;
}

void LNKFastArrayAddFastObject(LNKFastArrayRef array, LNKFastObjectRef object) {
	NSCAssert(array != NULL, @"The array must not be NULL");
	NSCAssert(object != NULL, @"The object must not be NULL");
	
	if (array->count == array->capacity) {
		array->capacity *= 2;
		array->parent.buffer = realloc(array->parent.buffer, array->capacity * sizeof(LNKFastObjectRef));
	}

	LNKFastObjectRef *const allObjects = array->parent.buffer;
	allObjects[array->count] = object;
	array->count++;

	LNKFastObjectRetain(object);
}

LNKFastObjectRef LNKFastArrayObjectAtIndex(LNKFastArrayRef array, LNKSize index) {
	NSCAssert(array != NULL, @"The array must not be NULL");
	NSCAssert(index < array->count, @"The index must be within bounds");

	LNKFastObjectRef *const allObjects = array->parent.buffer;
	return allObjects[index];
}

LNKSize LNKFastArrayObjectCount(LNKFastArrayRef array) {
	NSCAssert(array != NULL, @"The array must not be NULL");
	
	return array->count;
}

void LNKFastArrayRelease(LNKFastArrayRef array) {
	NSCAssert(array != NULL, @"The array must not be NULL");

	// this is about to go down to 0
	if (array->parent.retainCount == 1) {
		LNKFastObjectRef *const allObjects = array->parent.buffer;

		for (LNKSize i = 0; i < array->count; i++) {
			LNKFastObjectRelease(allObjects[i]);
		}
	}

	LNKFastObjectRelease((LNKFastObjectRef)array);
}

LNKFastStringRef LNKFastStringCreateWithUTF8String(const char *rawString, LNKSize length) {
	NSCAssert(rawString != NULL, @"The string must not be NULL");

	LNKFastStringRef string = calloc(sizeof(LNKFastString), 1);
	string->parent.retainCount = 1;
	string->parent.buffer = calloc(length + 1, sizeof(char));
	strncpy((char *)string->parent.buffer, rawString, length);
	return string;
}

const char *LNKFastStringGetUTF8String(LNKFastStringRef string) {
	NSCAssert(string != NULL, @"The string must not be NULL");
	return (const char *)string->parent.buffer;
}
