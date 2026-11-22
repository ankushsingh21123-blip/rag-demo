BEGIN;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;
CREATE SCHEMA IF NOT EXISTS semantic_atlas_private;
REVOKE ALL ON SCHEMA semantic_atlas_private FROM PUBLIC, anon, authenticated;
CREATE TABLE IF NOT EXISTS semantic_atlas_private.passages (
 id integer PRIMARY KEY,
 title text NOT NULL CHECK (length(title) <= 120),
 topic text NOT NULL CHECK (length(topic) <= 40),
 content text NOT NULL CHECK (length(content) <= 2000),
 embedding extensions.vector(384) NOT NULL
);
ALTER TABLE semantic_atlas_private.passages ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON semantic_atlas_private.passages FROM PUBLIC, anon, authenticated;
CREATE OR REPLACE FUNCTION public.semantic_atlas_search(query_embedding extensions.vector, result_limit integer DEFAULT 3)
RETURNS TABLE(id integer,title text,topic text,content text,score double precision)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = '' SET statement_timeout = '1000ms'
AS $$
BEGIN
 IF extensions.vector_dims(query_embedding) <> 384 OR extensions.vector_norm(query_embedding) < 0.98 OR extensions.vector_norm(query_embedding) > 1.02 THEN
  RAISE EXCEPTION 'Normalised 384-dimensional vector required';
 END IF;
 RETURN QUERY SELECT p.id,p.title,p.topic,p.content,
  (1 - (p.embedding OPERATOR(extensions.<=>) query_embedding))::double precision
 FROM semantic_atlas_private.passages p
 ORDER BY p.embedding OPERATOR(extensions.<=>) query_embedding
 LIMIT least(greatest(coalesce(result_limit,3),1),5);
END;
$$;
REVOKE ALL ON FUNCTION public.semantic_atlas_search(extensions.vector,integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.semantic_atlas_search(extensions.vector,integer) TO anon,authenticated;
CREATE OR REPLACE FUNCTION public.semantic_atlas_health()
RETURNS jsonb LANGUAGE sql STABLE SECURITY DEFINER SET search_path = '' SET statement_timeout = '1000ms'
AS $$ SELECT jsonb_build_object('corpus','fictional support documentation','documents',count(*),'dimensions',384,'storage','PostgreSQL pgvector','access','read-only RPC') FROM semantic_atlas_private.passages $$;
REVOKE ALL ON FUNCTION public.semantic_atlas_health() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.semantic_atlas_health() TO anon,authenticated;
COMMIT;
