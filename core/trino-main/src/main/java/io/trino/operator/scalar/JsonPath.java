/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.trino.operator.scalar;

import io.airlift.slice.Slice;

import static java.util.Objects.requireNonNull;

public class JsonPath
{
    private final String pattern;
    private final JsonExtract.JsonExtractor<Slice> scalarExtractor;
    private final JsonExtract.JsonExtractor<Slice> objectExtractor;
    private final JsonExtract.JsonExtractor<Long> sizeExtractor;

    public JsonPath(String pattern)
    {
        this.pattern = requireNonNull(pattern, "pattern is null");
        scalarExtractor = JsonExtract.generateExtractor(pattern, new JsonExtract.ScalarValueJsonExtractor());
        objectExtractor = JsonExtract.generateExtractor(pattern, new JsonExtract.JsonValueJsonExtractor());
        sizeExtractor = JsonExtract.generateExtractor(pattern, new JsonExtract.JsonSizeExtractor());
    }

    public String pattern()
    {
        return pattern;
    }

    public JsonExtract.JsonExtractor<Slice> getScalarExtractor()
    {
        return scalarExtractor;
    }

    public JsonExtract.JsonExtractor<Slice> getObjectExtractor()
    {
        return objectExtractor;
    }

    public JsonExtract.JsonExtractor<Long> getSizeExtractor()
    {
        return sizeExtractor;
    }

    @Override
    public String toString()
    {
        return pattern;
    }
}
