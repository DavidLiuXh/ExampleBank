package com.mytest;

import com.google.auto.value.AutoValue;

@AutoValue
public abstract class Animal {
    static Animal create(String name) {
        return new AutoValue_Animal(name);
    }

    public abstract String name();
}
